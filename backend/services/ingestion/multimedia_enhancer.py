import os
import re
import json
import logging
import asyncio
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

VIDEO_EVENT_PROMPT = """你是多模态知识提取专家。以下是一个视频片段的转写内容和画面描述，请从中提取结构化知识点。

## 输入格式
- [时间戳] 说话人: 台词
- [画面描述] 视觉内容描述

## 输出要求
只输出一个JSON数组，每个元素包含：
{{
    "fact": "确切的陈述（主语+谓语+宾语+时间+地点+数值）",
    "category": "概念/事实/方法/观点/待验证",
    "confidence": 0.0-1.0,
    "related_entities": ["实体列表"],
    "evidence_type": "video",
    "evidence_ref": "时间戳范围",
    "speaker": "说话人标签",
    "scene_description": "相关画面描述"
}}

## 思维链
1. 先列出观察到的内容要素（人物、动作、数据、论述）
2. 再推理出可用作证据的知识点
3. 为每条知识评估可信度

输入内容：
{content}"""

AUDIO_QA_PROMPT = """你是对话/讲座内容分析师。从以下音频转录稿中提取结构化的问答对。

## 输出要求
只输出一个JSON数组：
[{{
    "question": "提出的问题",
    "answer": "给出的回答",
    "topic": "主题标签",
    "confidence": 0.0-1.0,
    "speaker_role": "提问者/回答者"
}}]

输入内容：
{content}"""

AUDIO_SEGMENT_SUMMARY_PROMPT = """你是音频内容分析专家。为以下音频片段生成结构化摘要。

## 输出要求
只输出一个JSON对象：
{{
    "title": "片段标题（10字以内）",
    "key_points": ["要点1", "要点2", "要点3"],
    "sentiment": "正面/负面/中性/激昂/平和",
    "intent": "陈述/提问/辩论/讲解/闲聊/指示",
    "keywords": ["关键词1", "关键词2"],
    "action_items": ["需要执行的行动项"]
}}

输入内容：
{content}"""

IMAGE_MULTI_ANGLE_PROMPT = """你是图像多维度理解专家。请从以下OCR文字和图像描述中提取知识。

## 输出要求
只输出一个JSON数组，每个元素包含：
{{
    "fact": "确切的陈述",
    "category": "概念/事实/方法/观点/待验证",
    "confidence": 0.0-1.0,
    "evidence_type": "image",
    "image_element": "关联的图像元素（文字/图表/流程图/照片）",
    "relations": ["与图像中其他元素的关系描述"]
}}

## 思维链
1. 观察：列出图中可见的所有元素（文字块、图表、人物、物体、流程）
2. 理解：推断元素之间的逻辑关系
3. 提取：生成可推理的结构化知识点

输入内容：
{content}"""

IMAGE_CHART_PROMPT = """你是图表数据分析师。从以下图表描述中提取结构化数据知识。

## 输出要求
只输出一个JSON对象：
{{
    "chart_type": "柱状图/饼图/折线图/流程图/思维导图/组织结构图/其他",
    "title": "图表标题",
    "data_points": [{{"label": "标签", "value": "数值", "unit": "单位"}}],
    "trend": "上升/下降/平稳/波动",
    "key_finding": "核心发现（一句话）",
    "facts": ["从图表中提取的知识点列表"],
    "confidence": 0.0-1.0
}}

输入内容：
{content}"""


class MultimediaEnhancer:
    def __init__(self, settings=None):
        self.settings = settings
        self._deepseek_client = None
        self._vision_client = None
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    def _get_deepseek(self):
        if self._deepseek_client is None and self.settings:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
            )
        return self._deepseek_client

    def _get_vision_llm(self):
        if self.settings is None or not getattr(self.settings, 'vision_enabled', False):
            return None
        if self._vision_client is None:
            from openai import AsyncOpenAI
            api_key = self.settings.vision_api_key or self.settings.deepseek_api_key
            self._vision_client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.settings.vision_base_url,
            )
        return self._vision_client

    async def _call_deepseek(self, prompt: str, max_tokens: int = 2048) -> str:
        client = self._get_deepseek()
        if not client:
            return ""
        try:
            response = await client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"DeepSeek API 调用失败: {e}")
            return ""

    # ================================================================
    #  视频增强
    # ================================================================

    def detect_scenes(self, file_path: str, threshold: float = 0.35) -> list[dict]:
        scene_change_times = []
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-i", file_path,
                "-vf", f"select='gt(scene\\\\,{threshold})',showinfo",
                "-vsync", "vfr", "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            time_pattern = re.compile(r"pts_time:(\d+\.?\d*)")
            for line in result.stderr.split("\n"):
                match = time_pattern.search(line)
                if match:
                    scene_change_times.append(float(match.group(1)))
        except Exception as e:
            logger.warning(f"场景检测失败: {e}")

        if not scene_change_times:
            scene_change_times = [0.0]
        return sorted(set(round(t, 1) for t in scene_change_times))

    def extract_keyframes(self, file_path: str, times: list[float], output_dir: str) -> list[str]:
        if not times:
            return []
        frame_paths = []
        try:
            import subprocess
            os.makedirs(output_dir, exist_ok=True)
            for i, t in enumerate(times[:20]):
                frame_path = os.path.join(output_dir, f"scene_{i:03d}_{int(t)}s.jpg")
                cmd = [
                    "ffmpeg", "-ss", str(t), "-i", file_path,
                    "-vframes", "1", "-q:v", "2", frame_path, "-y"
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    frame_paths.append(frame_path)
        except Exception as e:
            logger.warning(f"关键帧提取失败: {e}")
        return frame_paths

    def describe_frames(self, frame_paths: list[str]) -> list[str]:
        vision_client = self._get_vision_llm()
        if vision_client:
            return self._describe_frames_vision(frame_paths, vision_client)
        return self._describe_frames_basic(frame_paths)

    def _describe_frames_vision(self, frame_paths: list[str], vision_client) -> list[str]:
        import asyncio, base64

        async def _describe_one(fp):
            try:
                with open(fp, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                response = await vision_client.chat.completions.create(
                    model=self.settings.vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请用1-3句话描述这张视频帧的画面内容，包括：场景类型、主要物体/人物、正在发生的动作。只返回纯文本描述，不要JSON。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    }],
                    temperature=0.1,
                    max_tokens=200,
                )
                desc = response.choices[0].message.content or ""
                ocr_text = self._ocr_frame(fp)
                if ocr_text:
                    return f"[画面描述] {desc.strip()}\n[画面文字] {ocr_text[:500]}"
                return f"[画面描述] {desc.strip()}"
            except Exception as e:
                logger.warning(f"[视频帧] 视觉模型描述失败: {e}")
                return self._describe_frame_basic(fp)

        async def _describe_all():
            tasks = [_describe_one(fp) for fp in frame_paths]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            return asyncio.run(_describe_all())
        except RuntimeError:
            return asyncio.run(_describe_all())

    def _describe_frame_basic(self, frame_path: str) -> str:
        ocr_text = self._ocr_frame(frame_path)
        basic = self._basic_frame_analysis(frame_path)
        if ocr_text:
            return f"[画面文字] {ocr_text[:500]}\n{basic}"
        return basic

    def _describe_frames_basic(self, frame_paths: list[str]) -> list[str]:
        return [self._describe_frame_basic(fp) for fp in frame_paths]

    def _ocr_frame(self, frame_path: str) -> str:
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(frame_path)
            for lang in ("chi_sim+eng", "eng"):
                try:
                    text = pytesseract.image_to_string(img, lang=lang)
                    if text.strip():
                        return text.strip()
                except Exception:
                    continue
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"关键帧OCR失败: {e}")
        return ""

    def _basic_frame_analysis(self, frame_path: str) -> str:
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(frame_path).convert("RGB")
            arr = np.array(img)
            h, w, _ = arr.shape
            gray = np.mean(arr, axis=2)
            brightness = np.mean(gray)
            text_density = np.sum(gray < 100) / (h * w)
            parts = [f"[画面分析] 分辨率: {w}x{h}"]
            if text_density > 0.05:
                parts.append("文字密集型画面（可能是PPT/文档）")
            elif text_density > 0.02:
                parts.append("混合型画面（含部分文字和图形）")
            else:
                parts.append("视觉型画面（照片/图表为主）")
            channel_max = np.max(arr, axis=(0, 1))
            dominant = []
            if channel_max[0] > 200:
                dominant.append("偏红")
            if channel_max[1] > 200:
                dominant.append("偏绿")
            if channel_max[2] > 200:
                dominant.append("偏蓝")
            if dominant:
                parts.append(f"主色调: {'+'.join(dominant)}")
            return "\n".join(parts)
        except Exception:
            return "[画面分析] 无法解析"

    def bind_scene_events(
        self, segments: list[dict], scene_times: list[float],
        frame_descriptions: list[str]
    ) -> list[dict]:
        events = []
        scene_boundaries = scene_times + [float("inf")]
        scene_idx = 0

        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_mid = (seg_start + seg_end) / 2
            seg_text = seg.get("text", "").strip()

            while scene_idx < len(scene_boundaries) - 1 and seg_mid >= scene_boundaries[scene_idx + 1]:
                scene_idx += 1

            desc = frame_descriptions[scene_idx] if scene_idx < len(frame_descriptions) else ""
            events.append({
                "start": seg_start,
                "end": seg_end,
                "mid": seg_mid,
                "text": seg_text,
                "scene_index": scene_idx,
                "scene_time": scene_times[scene_idx] if scene_idx < len(scene_times) else 0,
                "frame_description": desc,
            })
        return events

    async def extract_video_knowledge(self, events: list[dict], source: str) -> list[dict]:
        results = []
        chunk_size = 5
        for batch_start in range(0, len(events), chunk_size):
            batch = events[batch_start:batch_start + chunk_size]
            content_parts = []
            for e in batch:
                ts = f"[{int(e['start']//60):02d}:{int(e['start']%60):02d}-{int(e['end']//60):02d}:{int(e['end']%60):02d}]"
                content_parts.append(f"{ts} {e['text']}")
                if e.get("frame_description"):
                    content_parts.append(f"[画面描述] {e['frame_description'][:300]}")
            content = "\n".join(content_parts)
            if len(content) < 20:
                continue

            resp = await self._call_deepseek(
                VIDEO_EVENT_PROMPT.format(content=content[:3000]), max_tokens=2048
            )
            if resp:
                try:
                    match = re.search(r"\[.*\]", resp, re.DOTALL)
                    if match:
                        items = json.loads(match.group())
                        for item in items:
                            item["source"] = source
                        results.extend(items)
                except (json.JSONDecodeError, KeyError):
                    pass
        return results

    # ================================================================
    #  音频增强
    # ================================================================

    def smart_segment(self, segments: list[dict], pause_threshold: float = 2.0,
                      max_segment_len: int = 60) -> list[list[dict]]:
        if not segments:
            return []
        groups = []
        current = [segments[0]]
        current_len = segments[0].get("end", 0) - segments[0].get("start", 0)

        for i in range(1, len(segments)):
            seg = segments[i]
            prev = segments[i - 1]
            gap = seg.get("start", 0) - prev.get("end", 0)
            seg_dur = seg.get("end", 0) - seg.get("start", 0)

            if gap > pause_threshold or current_len + seg_dur > max_segment_len:
                groups.append(current)
                current = [seg]
                current_len = seg_dur
            else:
                current.append(seg)
                current_len += seg_dur

        if current:
            groups.append(current)
        return groups

    async def generate_segment_summary(self, seg_group: list[dict]) -> dict:
        text = " ".join(s.get("text", "") for s in seg_group)
        if len(text) < 10:
            return {"title": "短片段", "key_points": [text[:50]], "sentiment": "中性", "intent": "陈述", "keywords": []}

        resp = await self._call_deepseek(
            AUDIO_SEGMENT_SUMMARY_PROMPT.format(content=text[:2000]), max_tokens=1024
        )
        if resp:
            try:
                match = re.search(r"\{.*\}", resp, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"title": f"片段", "key_points": [text[:100]], "sentiment": "中性", "intent": "陈述", "keywords": []}

    async def extract_qa_pairs(self, full_text: str) -> list[dict]:
        if len(full_text) < 30:
            return []
        resp = await self._call_deepseek(
            AUDIO_QA_PROMPT.format(content=full_text[:2500]), max_tokens=2048
        )
        if resp:
            try:
                match = re.search(r"\[.*\]", resp, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    def extract_key_sentences(self, text: str, top_k: int = 5) -> list[str]:
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text) if text else []
        except Exception:
            sentences = re.split(r"[。！？.!?\n]+", text)

        if len(sentences) <= top_k:
            return [s.strip() for s in sentences if len(s.strip()) > 5]

        try:
            from keybert import KeyBERT
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(text, top_n=top_k * 2)
            keyword_set = set(k[0] for k in keywords)
            scored = []
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:
                    continue
                score = sum(1 for kw in keyword_set if kw.lower() in sent.lower())
                scored.append((score, sent))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s[1] for s in scored[:top_k]]
        except Exception:
            return [s.strip() for s in sentences[:top_k] if len(s.strip()) > 5]

    def detect_emotion(self, text: str) -> str:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.5:
                return "积极"
            elif polarity > 0.1:
                return "正面"
            elif polarity < -0.5:
                return "消极"
            elif polarity < -0.1:
                return "负面"
            else:
                pass
        except Exception:
            pass

        emotion_keywords = {
            "疑问": ["为什么", "怎么", "如何", "吗", "呢", "什么"],
            "肯定": ["一定", "确实", "肯定", "必然", "绝对"],
            "愤怒": ["气死", "过分", "怒", "火", "混蛋"],
            "犹豫": ["可能", "也许", "大概", "好像", "似乎"],
            "激昂": ["加油", "冲", "奋斗", "必胜", "向前"],
            "悲伤": ["难过", "伤心", "遗憾", "失败", "失去"],
        }
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text for kw in keywords):
                return emotion
        return "中性"

    def detect_intent(self, text: str) -> str:
        intent_keywords = {
            "提问": ["?", "？", "请问", "问一下", "怎么"],
            "讲解": ["所以", "因此", "因为", "首先", "然后", "总结"],
            "指示": ["必须", "应该", "需要", "不要", "请勿"],
            "闲聊": ["哈哈", "嗯嗯", "好的呀", "拜拜"],
            "辩论": ["不对", "但是", "然而", "相反", "我不同意"],
        }
        scores = {}
        for intent, keywords in intent_keywords.items():
            scores[intent] = sum(1 for kw in keywords if kw in text)
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        return "陈述"

    # ================================================================
    #  图片增强
    # ================================================================

    def layered_image_understanding(self, file_path: str) -> dict:
        result = {"ocr_text": "", "visual_analysis": "", "chart_data": None, "relations": []}
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(file_path)
            arr = np.array(img.convert("RGB"))
            h, w, _ = arr.shape
            result["width"] = w
            result["height"] = h
            result["aspect_ratio"] = round(w / h, 2)

            gray = np.mean(arr, axis=2)
            text_density = np.sum(gray < 100) / (h * w)
            result["text_density"] = round(text_density, 3)
            result["brightness"] = round(float(np.mean(gray)), 1)

            std_r, std_g, std_b = [float(np.std(arr[:,:,c])) for c in range(3)]
            if std_r < 15 and std_g < 15 and std_b < 15:
                result["visual_type"] = "低对比度文档"
            elif text_density > 0.08:
                result["visual_type"] = "文字密集型（文档/PPT/截图）"
            elif text_density > 0.03:
                result["visual_type"] = "混合型（图文结合）"
            elif std_r > 60 or std_g > 60 or std_b > 60:
                result["visual_type"] = "色彩丰富型（照片/设计图）"
            else:
                result["visual_type"] = "图形/图表型"
        except Exception as e:
            result["error"] = str(e)
        return result

    async def extract_chart_data(self, image_desc: str, ocr_text: str) -> dict:
        content = f"OCR文本:\n{ocr_text[:500]}\n\n图像描述:\n{image_desc}"
        resp = await self._call_deepseek(
            IMAGE_CHART_PROMPT.format(content=content[:2000]), max_tokens=2048
        )
        if resp:
            try:
                match = re.search(r"\{.*\}", resp, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"chart_type": "未知", "facts": [], "data_points": [], "confidence": 0.3}

    async def extract_image_knowledge(self, ocr_text: str, visual_analysis: dict, source: str) -> list[dict]:
        content = f"OCR文字:\n{ocr_text[:1000]}\n\n图像信息: 类型={visual_analysis.get('visual_type','')} 分辨率={visual_analysis.get('width','')}x{visual_analysis.get('height','')}"
        if len(content) < 50:
            return []
        resp = await self._call_deepseek(
            IMAGE_MULTI_ANGLE_PROMPT.format(content=content[:2500]), max_tokens=2048
        )
        results = []
        if resp:
            try:
                match = re.search(r"\[.*\]", resp, re.DOTALL)
                if match:
                    items = json.loads(match.group())
                    for item in items:
                        item["source"] = source
                    results = items
            except (json.JSONDecodeError, KeyError):
                pass
        return results