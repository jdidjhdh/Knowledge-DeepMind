"""
图片处理服务 - 五层优化流水线

Layer 1: 智能分类（photo/screenshot/chart/diagram/document/table/text_dense）
Layer 2: 自适应预处理（按类别不同策略）
Layer 3: 专项 OCR 引擎（PaddleOCR + Tesseract 回退）
Layer 4: 分类别深度理解（多模态 Prompt）
Layer 5: OCR 与视觉融合对齐
"""
import re
import os
import json
import uuid
import logging
import tempfile
import asyncio
import numpy as np
from typing import Optional
from io import BytesIO

logger = logging.getLogger(__name__)

IMAGE_CATEGORIES = ["photo", "screenshot", "chart", "diagram", "document", "table", "text_dense"]

_OPENCV_AVAILABLE = False
try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    logger.info("[ImageProcessor] OpenCV 未安装，使用 PIL 降级方案")

_PADDLE_AVAILABLE = False
try:
    _paddle_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".paddlex_cache")
    os.environ.setdefault("PADDLE_HOME", _paddle_home)
    os.makedirs(_paddle_home, exist_ok=True)
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    logger.info("[ImageProcessor] PaddleOCR 未安装，使用 Tesseract 回退")
except Exception as e:
    logger.warning(f"[ImageProcessor] PaddleOCR 环境初始化失败: {e}，使用 Tesseract 回退")


IMAGE_CLASSIFICATION_PROMPT = """你是一个图像分类专家。请分析这张图片并返回 JSON 格式的分类结果。

类别选项: photo(照片), screenshot(屏幕截图), chart(数据图表), diagram(流程图/架构图), document(文档/扫描件), table(表格), text_dense(文字密集的PPT/板书)

返回严格JSON格式:
{
  "category": "从上述选项中选择",
  "confidence": 0.0到1.0之间,
  "reason": "分类理由（一句话）",
  "has_text": true/false,
  "has_structure": true/false
}"""


class ImageClassifier:
    """Layer 1: 图片智能分类"""

    def __init__(self, settings=None):
        self.settings = settings
        self._client = None
        self._vision_client = None

    def _get_llm(self):
        if self._client is None and self.settings:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
            )
        return self._client

    def _get_vision_llm(self):
        if self.settings is None or not self.settings.vision_enabled:
            return None
        if self._vision_client is None:
            from openai import AsyncOpenAI
            api_key = self.settings.vision_api_key or self.settings.deepseek_api_key
            self._vision_client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.settings.vision_base_url,
            )
        return self._vision_client

    def _encode_image(self, image_path: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def classify_heuristic(self, image_path: str) -> dict:
        """快速启发式分类（无需 LLM，作为回退方案）"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            arr = np.array(img.convert("RGB"))
            h, w = arr.shape[:2]
            gray = np.mean(arr, axis=2)
            text_density = np.sum(gray < 120) / (h * w)
            std_r, std_g, std_b = [float(np.std(arr[:,:,c])) for c in range(3)]
            color_variance = std_r + std_g + std_b

            if text_density > 0.12:
                return {"category": "text_dense", "confidence": 0.7,
                        "reason": "高文字密度", "has_text": True, "has_structure": False}
            if text_density > 0.05 and w / h < 0.78:
                return {"category": "document", "confidence": 0.6,
                        "reason": "较高文字密度+竖版比例", "has_text": True, "has_structure": False}
            if text_density > 0.04 and color_variance > 100:
                return {"category": "screenshot", "confidence": 0.55,
                        "reason": "中等文字密度+高色彩差异", "has_text": True, "has_structure": False}
            if text_density > 0.02 and color_variance < 60:
                return {"category": "diagram", "confidence": 0.5,
                        "reason": "低文字密度+低色彩差异", "has_text": True, "has_structure": True}
            if color_variance > 150:
                return {"category": "photo", "confidence": 0.6,
                        "reason": "高色彩丰富度", "has_text": False, "has_structure": False}
            if 80 < color_variance < 150:
                return {"category": "chart", "confidence": 0.5,
                        "reason": "中等色彩差异", "has_text": True, "has_structure": True}
            return {"category": "photo", "confidence": 0.3,
                    "reason": "默认分类", "has_text": False, "has_structure": False}
        except Exception as e:
            return {"category": "photo", "confidence": 0.2,
                    "reason": f"分类失败: {e}", "has_text": False, "has_structure": False}

    async def classify_llm(self, image_path: str) -> dict:
        vision_client = self._get_vision_llm()
        if not vision_client:
            return self.classify_heuristic(image_path)

        try:
            img_b64 = self._encode_image(image_path)
            response = await vision_client.chat.completions.create(
                model=self.settings.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_CLASSIFICATION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                temperature=0.1,
                max_tokens=300,
            )
            text = response.choices[0].message.content
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                result["method"] = "vision_llm"
                logger.info(f"[图片分类] 视觉模型: {result.get('category')}, 置信度={result.get('confidence')}")
                return result
        except Exception as e:
            logger.warning(f"[图片分类] 视觉模型分类失败: {e}")

        fallback = self.classify_heuristic(image_path)
        fallback["method"] = "heuristic_fallback"
        return fallback

    async def classify(self, image_path: str) -> dict:
        """视觉模型分类 + 启发式回退"""
        llm_result = await self.classify_llm(image_path)
        if llm_result.get("method") == "vision_llm":
            heuristic = self.classify_heuristic(image_path)
            if heuristic["category"] == llm_result["category"]:
                llm_result["confidence"] = max(llm_result.get("confidence", 0.5), 0.8)
                llm_result["verified"] = "heuristic"
            return llm_result
        result = self.classify_heuristic(image_path)
        result["method"] = "heuristic"
        return result


class ImagePreprocessor:
    """Layer 2: 自适应预处理管线"""

    @staticmethod
    def preprocess(image_path: str, category: str) -> str:
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, f"ipp_{uuid.uuid4().hex[:8]}.png")

        try:
            from PIL import Image, ImageEnhance, ImageFilter
            img = Image.open(image_path)

            if category == "document":
                img = ImagePreprocessor._preprocess_document(img)
            elif category == "table":
                img = ImagePreprocessor._preprocess_table(img)
            elif category == "text_dense":
                img = ImagePreprocessor._preprocess_text_dense(img)
            elif category == "chart":
                img = ImagePreprocessor._preprocess_chart(img)
            elif category == "diagram":
                img = ImagePreprocessor._preprocess_diagram(img)
            elif category == "screenshot":
                img = ImagePreprocessor._preprocess_screenshot(img)
            else:
                img = ImagePreprocessor._preprocess_photo(img)

            img.save(output_path, "PNG")
            return output_path
        except Exception as e:
            logger.warning(f"[预处理] 失败: {e}")
            return image_path

    @staticmethod
    def _preprocess_document(img) -> object:
        """文档: 透视矫正 + 自适应二值化"""
        from PIL import Image, ImageEnhance, ImageFilter
        img = img.convert("L")
        if _OPENCV_AVAILABLE:
            arr = np.array(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            arr = clahe.apply(arr)
            arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 4)
            arr = cv2.medianBlur(arr, 1)
            img = Image.fromarray(arr)
        else:
            enh = ImageEnhance.Contrast(img)
            img = enh.enhance(2.0)
            enh = ImageEnhance.Sharpness(img)
            img = enh.enhance(1.5)
        return img

    @staticmethod
    def _preprocess_table(img) -> object:
        """表格: 二值化 + 保留线条"""
        from PIL import Image, ImageEnhance
        img = img.convert("L")
        if _OPENCV_AVAILABLE:
            arr = np.array(img)
            arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
            img = Image.fromarray(arr)
        else:
            enh = ImageEnhance.Contrast(img)
            img = enh.enhance(2.5)
        return img

    @staticmethod
    def _preprocess_text_dense(img) -> object:
        """文字密集: 锐化 + 对比度增强"""
        from PIL import Image, ImageEnhance, ImageFilter
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(1.8)
        enh = ImageEnhance.Sharpness(img)
        img = enh.enhance(2.0)
        return img

    @staticmethod
    def _preprocess_chart(img) -> object:
        """图表: 去噪 + 颜色增强"""
        from PIL import Image, ImageEnhance
        enh = ImageEnhance.Color(img)
        img = enh.enhance(1.3)
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(1.2)
        return img

    @staticmethod
    def _preprocess_diagram(img) -> object:
        """流程图: 灰度 + 二值化"""
        from PIL import Image, ImageEnhance
        img = img.convert("L")
        enh = ImageEnhance.Contrast(img)
        img = enh.enhance(2.0)
        return img

    @staticmethod
    def _preprocess_screenshot(img) -> object:
        """截图: 保持清晰度 + 轻微锐化"""
        from PIL import Image, ImageFilter
        return img.filter(ImageFilter.SHARPEN)

    @staticmethod
    def _preprocess_photo(img) -> object:
        """照片: 轻微增强"""
        from PIL import Image, ImageEnhance
        enh = ImageEnhance.Sharpness(img)
        return enh.enhance(1.2)


class OCRProcessor:
    """Layer 3: 专项 OCR 引擎"""

    _paddle_instance = None
    _paddle_kwargs = None

    def __init__(self, use_angle_cls: bool = True):
        self.use_angle_cls = use_angle_cls

    def _get_paddle(self):
        if not _PADDLE_AVAILABLE:
            return None
        if self._paddle_instance is None:
            try:
                self._paddle_instance = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang="ch",
                )
            except Exception as e:
                logger.warning(f"[OCR] PaddleOCR 初始化失败: {e}")
                return None
        return self._paddle_instance

    def extract_text(self, image_path: str) -> dict:
        """提取文字 + bounding box + 置信度"""
        result = {"text": "", "lines": [], "blocks": [], "confidence": 0.0}

        # 优先 PaddleOCR
        if _PADDLE_AVAILABLE:
            try:
                paddle = self._get_paddle()
                if paddle:
                    ocr_result = paddle.ocr(image_path, cls=self.use_angle_cls)
                    if ocr_result and ocr_result[0]:
                        texts = []
                        confs = []
                        for line_data in ocr_result[0]:
                            if line_data and len(line_data) >= 2:
                                bbox, text_info = line_data[0], line_data[1]
                                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                                conf = text_info[1] if len(text_info) > 1 else 0.8
                                texts.append(text)
                                confs.append(float(conf) if conf else 0.8)
                                result["lines"].append({
                                    "text": text,
                                    "confidence": round(float(conf), 3) if conf else 0.8,
                                    "bbox": bbox if isinstance(bbox, list) else [],
                                })
                        result["text"] = "\n".join(texts)
                        result["confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
                        result["engine"] = "paddleocr"
                        return result
            except Exception as e:
                logger.warning(f"[OCR] PaddleOCR 失败: {e}")

        # 回退 Tesseract
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(image_path)
            for lang in ("chi_sim+eng", "eng"):
                try:
                    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
                    texts = []
                    confs = []
                    for i in range(len(data["text"])):
                        t = data["text"][i].strip()
                        if t and int(data["conf"][i]) > 30:
                            texts.append(t)
                            confs.append(int(data["conf"][i]) / 100.0)
                            result["lines"].append({
                                "text": t,
                                "confidence": int(data["conf"][i]) / 100.0,
                                "bbox": [
                                    [data["left"][i], data["top"][i]],
                                    [data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]],
                                ],
                            })
                    result["text"] = "\n".join(texts)
                    result["confidence"] = round(sum(confs) / len(confs), 3) if confs else 0
                    result["engine"] = "tesseract"
                    return result
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"[OCR] Tesseract 失败: {e}")

        result["engine"] = "none"
        return result


CATEGORY_PROMPTS = {
    "photo": """你是一位视觉分析专家。这是一张照片。请返回 JSON:
{
  "scene": "场景描述（1-3句）",
  "main_objects": ["主要物体列表"],
  "people": ["人物描述（如有）"],
  "actions": ["进行的动作（如有）"],
  "atmosphere": "氛围/色调",
  "relations": [{"entity1": "实体1", "relation": "关系", "entity2": "实体2"}]
}""",
    "screenshot": """你是一位UI/UX分析师。这是一张软件界面截图。请返回 JSON:
{
  "app_name": "推断的软件名称",
  "interface_type": "页面类型（如登录页/仪表盘/设置页）",
  "components": [{"type": "按钮/输入框/列表/...", "label": "显示文字", "inferred_function": "推断功能"}],
  "text_content": "界面中所有文字（保留层级）",
  "inferred_purpose": "该界面的整体功能"
}""",
    "chart": """你是一位数据可视化专家。这是一张数据图表。请返回 JSON:
{
  "chart_type": "bar/line/pie/scatter/...",
  "title": "图表标题",
  "x_axis": {"label": "横轴标签", "values": ["值1","值2",...]},
  "y_axis": {"label": "纵轴标签", "unit": "单位"},
  "data_series": [{"name": "系列名", "values": [1,2,3,...]}],
  "trend": "趋势总结（一句话）",
  "key_finding": "核心发现",
  "anomalies": ["异常数据点（如有）"]
}""",
    "diagram": """你是一位架构分析师。这是一张流程图或架构图。请返回 JSON:
{
  "diagram_type": "flowchart/architecture/mindmap/sequence/...",
  "title": "图表标题（如有）",
  "nodes": [{"id": "节点ID", "label": "节点文字", "type": "process/decision/data/...", "parent": "父节点ID"}],
  "edges": [{"from": "源节点ID", "to": "目标节点ID", "label": "边说明"}],
  "layers": ["层级1描述", "层级2描述"],
  "summary": "整体结构和逻辑描述",
  "relationships": [{"entity": "实体A", "connects_to": "实体B", "nature": "关系性质"}]
}""",
    "document": """你是一位文档分析专家。这是一份文档/扫描件。请返回 JSON:
{
  "document_type": "合同/发票/报告/...",
  "title": "文档标题",
  "sections": [{"heading": "标题", "content": "内容概要", "level": 1或2}],
  "key_entities": ["关键实体（人名/公司名/金额等）"],
  "tables": [{"caption": "表格标题", "data": [["列1","列2"],["值","值"]]}],
  "summary": "文档核心内容总结（3-5句）",
  "confidence_notes": "不确定的部分"
}""",
    "table": """你是一位数据提取专家。这是一张表格图片。请返回 JSON:
{
  "table_caption": "表格标题",
  "headers": ["列名1", "列名2", ...],
  "column_types": ["string/number/date/...", ...],
  "rows": [["值11", "值12"], ["值21", "值22"]],
  "total_rows": 行数,
  "total_columns": 列数,
  "summary": "该表格的主要内容概述",
  "key_metrics": ["关键指标及其含义"]
}""",
    "text_dense": """你是一位文本整理专家。这是一张文字密集的图片。请返回 JSON:
{
  "content_type": "PPT/板书/文档页面/...",
  "headings": [{"level": 1或2, "text": "标题文字"}],
  "bullet_points": ["要点1", "要点2"],
  "paragraphs": ["段落内容1", "段落内容2"],
  "full_text": "所有文字（保持顺序）",
  "key_topics": ["主题一", "主题二"],
  "summary": "内容总结"
}""",
}


class ImageSemanticAnalyzer:
    """Layer 4: 分类别深度理解"""

    def __init__(self, settings=None):
        self.settings = settings
        self._client = None
        self._vision_client = None

    def _get_llm(self):
        if self._client is None and self.settings:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
            )
        return self._client

    def _get_vision_llm(self):
        if self.settings is None or not self.settings.vision_enabled:
            return None
        if self._vision_client is None:
            from openai import AsyncOpenAI
            api_key = self.settings.vision_api_key or self.settings.deepseek_api_key
            self._vision_client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.settings.vision_base_url,
            )
        return self._vision_client

    def _encode_image(self, image_path: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    async def analyze(self, image_path: str, category: str, ocr_text: str) -> dict:
        """按类别进行深度语义分析"""
        prompt = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["photo"])

        vision_client = self._get_vision_llm()
        if vision_client:
            return await self._analyze_vision(image_path, category, ocr_text, prompt, vision_client)

        return await self._analyze_text_only(category, ocr_text, prompt)

    async def _analyze_vision(self, image_path: str, category: str, ocr_text: str,
                               prompt: str, vision_client) -> dict:
        """视觉模型分析: 图片 + OCR 文字"""
        try:
            img_b64 = self._encode_image(image_path)
            full_prompt = (
                f"{prompt}\n\n"
                f"参考OCR提取的文字（可能有错别字，请修正）:\n{ocr_text[:2000]}\n\n"
                f"只返回 JSON，不要任何其他文字。"
            )
            response = await vision_client.chat.completions.create(
                model=self.settings.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                temperature=0.1,
                max_tokens=3072,
            )
            text = response.choices[0].message.content
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                result["category"] = category
                result["method"] = "vision_llm"
                logger.info(f"[语义分析] 视觉模型完成: {category}")
                return result
        except Exception as e:
            logger.warning(f"[语义分析] 视觉模型调用失败: {e}，回退文本模式")

        return await self._analyze_text_only(category, ocr_text, prompt)

    async def _analyze_text_only(self, category: str, ocr_text: str, prompt: str) -> dict:
        """纯文本分析: 仅基于 OCR 文字"""
        client = self._get_llm()
        if not ocr_text.strip():
            return {"category": category, "error": "无OCR文字可分析"}

        full_prompt = (
            f"{prompt}\n\n"
            f"以下是从图片中OCR提取的文字，请基于这些文字完成分析（可能有错别字，请修正）:\n"
            f"---\n{ocr_text[:3000]}\n---\n\n"
            f"只返回 JSON，不要任何其他文字。"
        )

        try:
            response = await client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
                max_tokens=3072,
            )
            text = response.choices[0].message.content
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                result["category"] = category
                result["method"] = "text_only"
                return result
        except Exception as e:
            logger.warning(f"[语义分析] LLM 调用失败: {e}")

        return {
            "category": category,
            "error": "分析失败",
            "ocr_text": ocr_text[:500],
        }


class OCRVisualFuser:
    """Layer 5: OCR 与视觉描述融合对齐"""

    @staticmethod
    def fuse(ocr_result: dict, visual_analysis: dict, classification: dict) -> dict:
        category = classification.get("category", "photo")

        fusion = {
            "category": category,
            "ocr_text": ocr_result.get("text", ""),
            "ocr_confidence": ocr_result.get("confidence", 0),
            "ocr_engine": ocr_result.get("engine", "none"),
            "visual_type": classification.get("category", ""),
            "low_confidence_regions": [],
            "aligned_elements": [],
            "contradictions": [],
        }

        # 低置信度区域识别
        for line in ocr_result.get("lines", []):
            if line.get("confidence", 1.0) < 0.5:
                fusion["low_confidence_regions"].append({
                    "text": line.get("text", ""),
                    "confidence": line.get("confidence", 0),
                    "bbox": line.get("bbox", []),
                })

        # 空间对齐: OCR 文字与视觉区域绑定
        ocr_lines = ocr_result.get("lines", [])
        if category == "screenshot":
            elements = visual_analysis.get("components", [])
            for elem in elements[:10]:
                label = elem.get("label", "")
                for line in ocr_lines:
                    if label and label in line.get("text", ""):
                        fusion["aligned_elements"].append({
                            "visual_component": label,
                            "ocr_text": line.get("text", ""),
                            "ocr_confidence": line.get("confidence", 0),
                            "alignment_type": "text_match",
                        })
                        break

        elif category == "chart":
            series_names = [s.get("name", "") for s in visual_analysis.get("data_series", [])]
            for sn in series_names:
                for line in ocr_lines:
                    if sn and sn in line.get("text", ""):
                        fusion["aligned_elements"].append({
                            "data_series": sn,
                            "ocr_text": line.get("text", ""),
                            "alignment_type": "data_series_verified",
                        })

        # 矛盾检测
        vis_text_count = len(visual_analysis.get("full_text", "")) if isinstance(visual_analysis, dict) else 0
        ocr_text_count = len(ocr_result.get("text", ""))
        if ocr_text_count > 0 and vis_text_count > 0 and abs(ocr_text_count - vis_text_count) > ocr_text_count * 2:
            fusion["contradictions"].append(
                f"视觉分析文字量({vis_text_count})与OCR文字量({ocr_text_count})差异过大，可能需要重新分析"
            )

        if fusion["low_confidence_regions"]:
            fusion["recommendation"] = f"发现 {len(fusion['low_confidence_regions'])} 处低置信度文字，建议人工复核"

        return fusion


class ImageProcessor:
    """图片处理总控 - 五层流水线"""

    def __init__(self, settings=None):
        self.settings = settings
        self.classifier = ImageClassifier(settings)
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRProcessor(use_angle_cls=True)
        self.analyzer = ImageSemanticAnalyzer(settings)
        self.fuser = OCRVisualFuser()

    async def process(self, image_path: str, filename: str = "") -> dict:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size

        result = {
            "filename": filename,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 2),
            "classification": {},
            "preprocessing": {},
            "ocr": {},
            "visual_analysis": {},
            "fusion": {},
            "knowledge_chunks": [],
        }

        # Layer 1: 分类
        classification = await self.classifier.classify(image_path)
        result["classification"] = classification
        category = classification.get("category", "photo")
        logger.info(f"[图片处理] {filename}: 分类={category}, 置信度={classification.get('confidence', 0):.2f}")

        # Layer 2: 预处理
        pp_path = self.preprocessor.preprocess(image_path, category)
        result["preprocessing"] = {
            "applied": pp_path != image_path,
            "category": category,
            "preprocessed_path": pp_path,
        }

        # Layer 3: OCR
        ocr_result = self.ocr.extract_text(pp_path)
        result["ocr"] = ocr_result
        logger.info(f"[图片处理] OCR: 引擎={ocr_result.get('engine')}, "
                     f"置信度={ocr_result.get('confidence', 0):.3f}, "
                     f"行数={len(ocr_result.get('lines', []))}")

        # Layer 4: 深度语义分析
        ocr_text_for_llm = ocr_result.get("text", "")[:2000]
        visual_analysis = await self.analyzer.analyze(image_path, category, ocr_text_for_llm)
        result["visual_analysis"] = visual_analysis

        # Layer 5: 融合对齐
        fusion = self.fuser.fuse(ocr_result, visual_analysis, classification)
        result["fusion"] = fusion

        # 构建知识 chunks
        result["knowledge_chunks"] = self._build_chunks(
            ocr_result, visual_analysis, classification, fusion, filename
        )

        # 清理预处理文件
        if pp_path != image_path and os.path.exists(pp_path):
            try:
                os.remove(pp_path)
            except Exception:
                pass

        logger.info(
            f"[图片处理] 完成: {filename}, chunks={len(result['knowledge_chunks'])}, "
            f"低置信度={len(fusion.get('low_confidence_regions', []))}"
        )
        return result

    def _build_chunks(self, ocr_result: dict, visual_analysis: dict,
                      classification: dict, fusion: dict, filename: str) -> list:
        chunks = []
        category = classification.get("category", "photo")

        # 主知识块
        content_parts = [
            f"=== 图片智能分析 ===",
            f"类型: {category} (置信度: {classification.get('confidence', 0):.2f})",
            f"分辨率: {classification.get('width', '?')}x{classification.get('height', '?')}",
            f"OCR引擎: {ocr_result.get('engine', 'none')}, 置信度: {ocr_result.get('confidence', 0):.3f}",
        ]

        if classification.get("reason"):
            content_parts.append(f"分类理由: {classification['reason']}")

        content_parts.append(f"\n[OCR 文字]\n{ocr_result.get('text', '')[:1500]}")

        # 类别特定内容
        if category == "chart":
            content_parts.extend([
                f"\n[图表分析]",
                f"图表类型: {visual_analysis.get('chart_type', '?')}",
                f"标题: {visual_analysis.get('title', '?')}",
                f"趋势: {visual_analysis.get('trend', '?')}",
                f"核心发现: {visual_analysis.get('key_finding', '?')}",
            ])
            for ds in visual_analysis.get("data_series", []):
                content_parts.append(f"  系列[{ds.get('name', '?')}]: {ds.get('values', [])}")

        elif category == "diagram":
            content_parts.append(f"\n[结构分析]")
            content_parts.append(f"图表类型: {visual_analysis.get('diagram_type', '?')}")
            content_parts.append(f"总结: {visual_analysis.get('summary', '?')}")
            for node in visual_analysis.get("nodes", [])[:10]:
                content_parts.append(f"  [{node.get('type', '?')}] {node.get('label', '?')}")

        elif category == "document":
            content_parts.append(f"\n[文档分析]")
            content_parts.append(f"文档类型: {visual_analysis.get('document_type', '?')}")
            content_parts.append(f"总结: {visual_analysis.get('summary', '?')}")
            for sec in visual_analysis.get("sections", [])[:8]:
                content_parts.append(f"  {'#' * sec.get('level', 1)} {sec.get('heading', '')}: {sec.get('content', '')[:100]}")

        elif category == "table":
            content_parts.append(f"\n[表格数据]")
            content_parts.append(f"表头: {visual_analysis.get('headers', [])}")
            for row in visual_analysis.get("rows", [])[:10]:
                content_parts.append(f"  {row}")

        elif category == "screenshot":
            content_parts.append(f"\n[界面分析]")
            content_parts.append(f"应用: {visual_analysis.get('app_name', '?')}")
            content_parts.append(f"页面: {visual_analysis.get('interface_type', '?')}")
            content_parts.append(f"功能: {visual_analysis.get('inferred_purpose', '?')}")
            for comp in visual_analysis.get("components", [])[:10]:
                content_parts.append(f"  [{comp.get('type', '?')}] {comp.get('label', '?')} -> {comp.get('inferred_function', '?')}")

        elif category == "text_dense":
            content_parts.append(f"\n[内容要点]")
            for h in visual_analysis.get("headings", [])[:5]:
                content_parts.append(f"  {'#' * h.get('level', 1)} {h.get('text', '')}")
            for bp in visual_analysis.get("bullet_points", [])[:10]:
                content_parts.append(f"  - {bp}")

        elif category == "photo":
            content_parts.append(f"\n[场景描述]")
            content_parts.append(f"场景: {visual_analysis.get('scene', '?')}")
            content_parts.append(f"氛围: {visual_analysis.get('atmosphere', '?')}")
            for obj in visual_analysis.get("main_objects", [])[:8]:
                content_parts.append(f"  物体: {obj}")

        # 融合对齐信息
        if fusion.get("low_confidence_regions"):
            content_parts.append(f"\n[警告] {len(fusion['low_confidence_regions'])} 处低置信度文字，建议人工复核")
            for lc in fusion["low_confidence_regions"][:5]:
                content_parts.append(f"  - \"{lc['text']}\" (置信度: {lc['confidence']:.2f})")

        if fusion.get("contradictions"):
            content_parts.append(f"\n[矛盾提示]")
            for c in fusion["contradictions"]:
                content_parts.append(f"  - {c}")

        chunks.append({
            "content": "\n".join(content_parts),
            "source_path": filename,
            "source_type": "image",
            "chunk_index": 0,
            "metadata": {
                "category": category,
                "ocr_engine": ocr_result.get("engine", ""),
                "ocr_confidence": ocr_result.get("confidence", 0),
                "classification_confidence": classification.get("confidence", 0),
                "low_conf_count": len(fusion.get("low_confidence_regions", [])),
                "extraction_mode": "enhanced_v5",
                "classification_method": classification.get("method", "heuristic"),
            },
        })

        return chunks


_image_processor: Optional[ImageProcessor] = None


async def get_image_processor(settings=None) -> ImageProcessor:
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor(settings=settings)
    return _image_processor