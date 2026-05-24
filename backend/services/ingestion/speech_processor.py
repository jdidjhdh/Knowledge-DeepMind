"""
语音处理服务 - 多层优化音频/视频语音识别准确率

分层架构:
  Layer 0: 音频预处理（降噪、归一化、重采样）
  Layer 1: 云端 ASR 转写（通义千问 Audio / 兼容 OpenAI 格式）
  Layer 2: VAD 分段处理
  Layer 3: 后处理纠错（术语映射、LLM上下文纠错、置信度过滤）
  Layer 4: 说话人分离 + 用户反馈闭环
"""
import os
import re
import json
import uuid
import logging
import asyncio
import tempfile
import subprocess
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DOMAIN_TERMS = [
    "Transformer", "BERT", "GPT", "LLM", "RAG", "知识图谱", "向量数据库",
    "嵌入向量", "注意力机制", "自注意力", "多头注意力", "残差连接",
    "层归一化", "前馈网络", "位置编码", "tokenizer", "分词器",
    "微调", "预训练", "零样本学习", "少样本学习", "提示工程",
    "Neo4j", "Milvus", "Chroma", "Pinecone", "LangChain", "LlamaIndex",
    "DeepSeek", "OpenAI", "Anthropic", "百川", "通义千问", "文心一言",
    "API", "SDK", "REST", "GraphQL", "gRPC", "WebSocket",
    "Python", "PyTorch", "TensorFlow", "scikit-learn", "pandas", "numpy",
    "Docker", "Kubernetes", "Podman", "CI/CD", "DevOps", "MLOps",
    "GPU", "TPU", "CUDA", "cuDNN", "ONNX", "TensorRT",
    "JSON", "YAML", "CSV", "Parquet", "Protobuf", "MessagePack",
    "OAuth", "JWT", "RBAC", "CORS", "HTTPS", "TLS",
]

TERM_CORRECTION_MAP = {
    "自然语言处理": "自然语言处理",
    "机器学习": "机器学习",
    "深度学习": "深度学习",
    "神经网络": "神经网络",
    "卷积": "卷积",
    "循环": "循环",
    "梯度下降": "梯度下降",
    "反向传播": "反向传播",
    "过拟合": "过拟合",
    "正则化": "正则化",
    "决策树": "决策树",
    "支持向量机": "支持向量机",
    "随机森林": "随机森林",
    "梯度提升": "梯度提升",
    "主成分分析": "主成分分析",
    "聚类": "聚类",
    "分类": "分类",
    "回归": "回归",
    "贝叶斯": "贝叶斯",
    "马尔可夫": "马尔可夫",
    "蒙特卡洛": "蒙特卡洛",
    "遗传算法": "遗传算法",
    "粒计算": "粒计算",
    "语义网络": "语义网络",
    "本体论": "本体论",
    "推理引擎": "推理引擎",
    "知识工程": "知识工程",
}


class AudioPreprocessor:
    """Layer 0: 音频预处理 - 降噪、归一化、重采样"""

    @staticmethod
    def preprocess(input_path: str, output_dir: Optional[str] = None) -> str:
        output_dir = output_dir or tempfile.gettempdir()
        output_name = f"pp_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(output_dir, output_name)

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", (
                "highpass=f=80,"
                "lowpass=f=8000,"
                "afftdn=nr=10:nf=-25:tn=1,"
                "loudnorm=I=-16:TP=-1.5:LRA=11,"
                "volume=1.2"
            ),
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"[音频预处理] 完成: {os.path.getsize(output_path)} bytes")
                return output_path
            logger.warning(f"[音频预处理] 输出为空: {result.stderr[:300]}")
        except Exception as e:
            logger.warning(f"[音频预处理] ffmpeg失败: {e}")

        return input_path

    @staticmethod
    def preprocess_basic(input_path: str, output_dir: Optional[str] = None) -> str:
        """轻量预处理：仅重采样+归一化"""
        output_dir = output_dir or tempfile.gettempdir()
        output_name = f"pp_basic_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(output_dir, output_name)

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass

        return input_path

    @staticmethod
    def detect_audio_quality(input_path: str) -> dict:
        """检测音频质量指标"""
        quality = {"duration": 0, "sample_rate": 0, "channels": 0, "bitrate": 0, "has_noise": False}
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", input_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.stdout:
                info = json.loads(result.stdout)
                fmt = info.get("format", {})
                quality["duration"] = float(fmt.get("duration", 0))
                quality["bitrate"] = int(fmt.get("bit_rate", 0)) // 1000
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        quality["sample_rate"] = int(stream.get("sample_rate", 0))
                        quality["channels"] = int(stream.get("channels", 0))
                        break
        except Exception as e:
            logger.debug(f"[质量检测] 失败: {e}")
        return quality


class VADSegmenter:
    """Layer 2: 基于能量的 VAD 分段器"""

    @staticmethod
    def segment_by_silence(input_path: str, min_silence_ms: int = 500,
                           silence_thresh_db: int = -35) -> list[dict]:
        segments = []
        try:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-af", f"silencedetect=n={silence_thresh_db}dB:d={min_silence_ms / 1000:.1f}",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            silence_ends = []
            for line in result.stderr.split("\n"):
                if "silence_end:" in line:
                    m = re.search(r"silence_end:\s*([\d.]+)", line)
                    if m:
                        silence_ends.append(float(m.group(1)))
                elif "silence_start:" in line:
                    pass

            start = 0.0
            for end in sorted(silence_ends):
                dur = end - start
                if dur > 1.0:
                    segments.append({"start": round(start, 2), "end": round(end, 2), "duration": round(dur, 2)})
                start = end

            cmd_dur = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", input_path]
            dur_result = subprocess.run(cmd_dur, capture_output=True, text=True, timeout=30)
            total_dur = float(dur_result.stdout.strip() or 0)
            if total_dur > 0 and start < total_dur - 2:
                segments.append({"start": round(start, 2), "end": round(total_dur, 2),
                                 "duration": round(total_dur - start, 2)})

        except Exception as e:
            logger.warning(f"[VAD分段] 失败: {e}")

        if not segments:
            try:
                cmd_dur = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", input_path]
                dur_result = subprocess.run(cmd_dur, capture_output=True, text=True, timeout=30)
                total_dur = float(dur_result.stdout.strip() or 0)
                segments = [{"start": 0.0, "end": round(total_dur, 2), "duration": round(total_dur, 2)}]
            except Exception:
                segments = [{"start": 0.0, "end": 60.0, "duration": 60.0}]

        return segments


class CloudASRTranscriber:
    """Layer 1: 云端 ASR 转写器 - 兼容 OpenAI Audio API 格式"""

    def __init__(self, settings=None):
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None and self.settings:
            from openai import OpenAI
            api_key = self.settings.speech_api_key or self.settings.vision_api_key or self.settings.deepseek_api_key
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.settings.speech_base_url,
            )
        return self._client

    def transcribe(self, audio_path: str, language: str = "zh") -> dict:
        """调用云端 ASR 转写"""
        if not self.settings or not getattr(self.settings, 'speech_enabled', True):
            logger.warning("[CloudASR] 语音识别已关闭")
            return {"text": "", "segments": []}
        client = self._get_client()
        if not client:
            logger.error("[CloudASR] 无可用 API 配置")
            return {"text": "", "segments": []}

        logger.info(f"[CloudASR] 开始转写: {audio_path}, model={self.settings.speech_model}")

        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=self.settings.speech_model,
                    file=f,
                    language=language,
                    response_format="verbose_json",
                    temperature=0.0,
                    timestamp_granularities=["segment"],
                )

            text = getattr(response, "text", "") or ""
            segments_raw = getattr(response, "segments", []) or []

            segments = []
            for seg in segments_raw:
                segments.append({
                    "id": getattr(seg, "id", len(segments)),
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", "").strip(),
                    "words": [
                        {"word": w.get("word", ""), "probability": w.get("probability", 0.9),
                         "start": w.get("start", 0), "end": w.get("end", 0)}
                        for w in (getattr(seg, "words", None) or [])
                    ],
                })

            logger.info(f"[CloudASR] 完成: {len(text)} 字, {len(segments)} 段")
            return {"text": text, "segments": segments}

        except Exception as e:
            logger.error(f"[CloudASR] 转写失败: {e}")
            return {"text": "", "segments": []}

    def transcribe_segment(self, audio_path: str, start: float, end: float,
                           language: str = "zh") -> dict:
        """按时间段转写"""
        temp_path = os.path.join(tempfile.gettempdir(), f"asr_{uuid.uuid4().hex[:8]}.wav")
        try:
            dur = end - start
            cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
                   "-i", audio_path, "-ar", "16000", "-ac", "1", temp_path]
            subprocess.run(cmd, capture_output=True, timeout=60)

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                result = self.transcribe(temp_path, language=language)
                result["_segment_start"] = start
                result["_segment_end"] = end
                return result
        except Exception as e:
            logger.warning(f"[CloudASR 分段] 失败: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        return {"text": "", "segments": [], "_segment_start": start, "_segment_end": end}


class SpeechCorrector:
    """Layer 3: 后处理纠错"""

    def __init__(self, settings=None):
        self.settings = settings
        self._client = None
        self.term_map = TERM_CORRECTION_MAP
        self._correction_feedback = []

    def _get_llm(self):
        if self._client is None and self.settings:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
            )
        return self._client

    def apply_term_correction(self, text: str) -> str:
        """术语映射纠错"""
        corrected = text
        for wrong, right in self.term_map.items():
            if wrong in corrected and wrong != right:
                corrected = corrected.replace(wrong, right)
                logger.debug(f"[术语纠错] '{wrong}' -> '{right}'")
        return corrected

    def extract_low_confidence_words(self, segments: list[dict],
                                     threshold: float = 0.5) -> list[dict]:
        """提取低置信度词汇"""
        low_conf = []
        for seg in segments:
            words = seg.get("words", [])
            for w in words:
                if w.get("probability", 1.0) < threshold:
                    low_conf.append({
                        "word": w.get("word", "").strip(),
                        "confidence": round(w.get("probability", 0), 2),
                        "start": w.get("start", 0),
                        "end": w.get("end", 0),
                    })
        return low_conf

    def build_confidence_report(self, segments: list[dict]) -> dict:
        """构建置信度报告"""
        all_probs = []
        for seg in segments:
            for w in seg.get("words", []):
                if "probability" in w:
                    all_probs.append(w["probability"])

        if not all_probs:
            return {"avg_confidence": 0.8, "low_count": 0, "total_words": 0}

        avg = sum(all_probs) / len(all_probs)
        low = sum(1 for p in all_probs if p < 0.5)
        return {
            "avg_confidence": round(avg, 3),
            "low_count": low,
            "total_words": len(all_probs),
            "low_ratio": round(low / len(all_probs), 3) if all_probs else 0,
        }

    async def llm_context_correct(self, text: str, domain_context: str = "") -> str:
        """使用 LLM 进行上下文纠错"""
        client = self._get_llm()
        if not client or len(text) < 50:
            return text

        prompt = (
            "你是语音识别后处理专家。以下文本来自语音转写，可能包含:\n"
            "1. 同音错别字（如'知识图谱'错识别为'芝士图谱'）\n"
            "2. 专业术语识别错误\n"
            "3. 中英文混读的拼接错误\n\n"
            "请修正这些错误，但要保持原文的整体结构和意思不变。"
            "只输出修正后的完整文本，不要添加任何解释。\n\n"
        )
        if domain_context:
            prompt += f"领域上下文: {domain_context}\n\n"
        prompt += f"需要修正的文本:\n{text[:3000]}"

        try:
            response = await client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=min(len(text) * 2, 4096),
            )
            corrected = response.choices[0].message.content or text
            if len(corrected) > len(text) * 0.3:
                return corrected
        except Exception as e:
            logger.warning(f"[LLM纠错] 失败: {e}")

        return text

    def add_correction_feedback(self, original: str, correction: str):
        """记录用户修正反馈"""
        self._correction_feedback.append({
            "original": original,
            "correction": correction,
        })
        self.term_map[original] = correction
        logger.info(f"[反馈学习] '{original}' -> '{correction}'")


class SpeechProcessor:
    """语音处理总控 - 编排所有处理层"""

    def __init__(self, settings=None):
        self.settings = settings
        self.preprocessor = AudioPreprocessor()
        self.transcriber = CloudASRTranscriber(settings)
        self.corrector = SpeechCorrector(settings)
        self.segmenter = VADSegmenter()
        self._diarization_pipeline = None

    async def process_file(self, file_path: str, language: str = "zh",
                           enable_diarization: bool = False) -> dict:
        """完整处理流水线"""
        result = {
            "file_path": file_path,
            "language": language,
            "preprocessing": {},
            "transcription": {},
            "correction": {},
            "quality": {},
            "full_text": "",
            "segments": [],
            "confidence_report": {},
            "low_confidence_words": [],
        }

        # Layer 0: 质量检测
        result["quality"] = self.preprocessor.detect_audio_quality(file_path)
        logger.info(f"[语音处理] 音频质量: {result['quality']}")

        # Layer 0: 音频预处理
        pp_path = self.preprocessor.preprocess(file_path)
        result["preprocessing"] = {
            "original_path": file_path,
            "preprocessed_path": pp_path,
            "applied": pp_path != file_path,
        }

        # Layer 2: VAD 分段
        need_vad = result["quality"].get("duration", 0) > 120
        if need_vad:
            vad_segments = self.segmenter.segment_by_silence(pp_path)
            logger.info(f"[VAD] 分段为 {len(vad_segments)} 段")

            all_segments = []
            full_text_parts = []
            for vs in vad_segments:
                seg_result = self.transcriber.transcribe_segment(
                    pp_path, vs["start"], vs["end"], language
                )
                seg_text = seg_result.get("text", "").strip()
                if seg_text:
                    full_text_parts.append(seg_text)
                for s in seg_result.get("segments", []):
                    s["start"] += vs["start"]
                    s["end"] += vs["start"]
                    all_segments.append(s)

            raw_text = " ".join(full_text_parts)
            raw_segments = all_segments
        else:
            # Layer 1: Whisper 转写
            transcribe_result = self.transcriber.transcribe(pp_path, language=language)
            raw_text = transcribe_result.get("text", "").strip()
            raw_segments = transcribe_result.get("segments", [])

        result["transcription"] = {
            "model": self.settings.speech_model,
            "raw_text": raw_text,
            "segment_count": len(raw_segments),
            "temperature": 0.0,
        }

        # Layer 3: 术语纠错
        term_corrected = self.corrector.apply_term_correction(raw_text)

        # Layer 3: 置信度分析
        result["confidence_report"] = self.corrector.build_confidence_report(raw_segments)
        result["low_confidence_words"] = self.corrector.extract_low_confidence_words(raw_segments)

        # Layer 3: LLM 上下文纠错
        domain_context = "专业技术讨论，涉及人工智能、机器学习、知识图谱、大语言模型"
        final_text = await self.corrector.llm_context_correct(term_corrected, domain_context)
        if not final_text or len(final_text) < 10:
            final_text = term_corrected

        result["correction"] = {
            "term_corrected": term_corrected != raw_text,
            "llm_corrected": final_text != term_corrected,
        }
        result["full_text"] = final_text
        result["segments"] = raw_segments

        # Layer 4: 说话人分离（可选）
        if enable_diarization:
            result["diarization"] = await self._run_diarization(pp_path)

        # 清理预处理文件
        if pp_path != file_path and os.path.exists(pp_path):
            try:
                os.remove(pp_path)
            except Exception:
                pass

        logger.info(
            f"[语音处理] 完成: 置信度={result['confidence_report'].get('avg_confidence', 'N/A')}, "
            f"低置信词={result['confidence_report'].get('low_count', 0)}/{result['confidence_report'].get('total_words', 0)}, "
            f"分段={len(raw_segments)}"
        )
        return result

    async def _run_diarization(self, audio_path: str) -> list[dict]:
        """Layer 4: 说话人分离"""
        try:
            from pyannote.audio import Pipeline
            if self._diarization_pipeline is None:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.environ.get("HUGGINGFACE_TOKEN", ""),
                )
            diarization = self._diarization_pipeline(audio_path)
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                })
            return speakers
        except ImportError:
            logger.debug("[说话人分离] pyannote 未安装，跳过")
        except Exception as e:
            logger.debug(f"[说话人分离] 失败: {e}")
        return []


speech_processor: Optional[SpeechProcessor] = None


async def get_speech_processor(settings=None, reset: bool = False) -> SpeechProcessor:
    global speech_processor
    if reset or speech_processor is None:
        speech_processor = SpeechProcessor(settings=settings)
    return speech_processor