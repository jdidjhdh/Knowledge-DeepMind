import json
import logging
import re
from datetime import datetime, timedelta
from openai import AsyncOpenAI
from config import get_settings

logger = logging.getLogger(__name__)

TIME_EXTRACTION_PROMPT = """从以下文本中提取所有事件发生的时间，包括绝对时间（如"2024年3月"、"2025-03-15"）和相对时间（如"上周五"、"去年春节后"、"本月"）。

对每个时间，输出包含：
- "original": 原始表述
- "standard_date": 标准日期（ISO 8601 格式 YYYY-MM-DD，若仅知年份则为 YYYY-01-01，仅知年月为 YYYY-MM-01）
- "precision": 精度（"year" / "month" / "day"）
- "confidence": 置信度 (0-1)

若无法确定任何时间，输出 null。

文本：
{text}

请以 JSON 数组格式返回，仅返回 JSON，不要有其他内容。
示例输出：```json
[{"original":"2024年3月","standard_date":"2024-03-01","precision":"month","confidence":0.95}]
```"""

RELATIVE_TIME_PROMPT = """已知参考日期（锚点）为 {anchor_date}。

请将以下相对时间表述转换为标准日期（ISO 8601 YYYY-MM-DD 格式）：
原始表述：{expression}

输出 JSON：
{{
  "standard_date": "YYYY-MM-DD",
  "precision": "day",
  "confidence": 0.8
}}

仅返回 JSON，不要有其他内容。若无法确定，返回 {{}}。"""


class TimeExtractionService:
    def __init__(self):
        settings = get_settings()
        self.client: AsyncOpenAI | None = None
        self.model: str = settings.deepseek_model
        if settings.deepseek_api_key:
            self.client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

    async def extract_event_times(self, text: str, anchor_date: str = "") -> list[dict]:
        if not self.client:
            return []
        try:
            prompt = TIME_EXTRACTION_PROMPT.format(text=text[:4000])
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            results = self._parse_json_response(content)
            return self._norm_event_times(results)
        except Exception as e:
            logger.warning(f"时间提取失败: {e}")
            return []

    async def resolve_relative_time(self, expression: str, anchor_date: str) -> dict | None:
        if not self.client:
            return self._resolve_relative_time_heuristic(expression, anchor_date)
        try:
            prompt = RELATIVE_TIME_PROMPT.format(
                anchor_date=anchor_date or datetime.now().strftime("%Y-%m-%d"),
                expression=expression,
            )
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            result = self._parse_json_response(content)
            if isinstance(result, dict) and result.get("standard_date"):
                return result
            return None
        except Exception as e:
            logger.warning(f"相对时间解析失败: {e}")
            return self._resolve_relative_time_heuristic(expression, anchor_date)

    def _resolve_relative_time_heuristic(self, expression: str, anchor: str) -> dict | None:
        anchor_date = datetime.now()
        if anchor:
            try:
                anchor_date = datetime.strptime(anchor[:10], "%Y-%m-%d")
            except ValueError:
                pass
        exp = expression.strip()
        day_names = {"周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
                     "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6}
        for name, offset in day_names.items():
            if name in exp:
                target = anchor_date - timedelta(days=(anchor_date.weekday() - offset) % 7)
                if "上" in exp:
                    target -= timedelta(days=7)
                elif "下" in exp:
                    target += timedelta(days=7)
                return {"standard_date": target.strftime("%Y-%m-%d"), "precision": "day", "confidence": 0.6}
        if "昨天" in exp:
            target = anchor_date - timedelta(days=1)
            return {"standard_date": target.strftime("%Y-%m-%d"), "precision": "day", "confidence": 0.8}
        if "今天" in exp:
            return {"standard_date": anchor_date.strftime("%Y-%m-%d"), "precision": "day", "confidence": 0.9}
        if "明天" in exp:
            target = anchor_date + timedelta(days=1)
            return {"standard_date": target.strftime("%Y-%m-%d"), "precision": "day", "confidence": 0.8}
        if "本周" in exp:
            monday = anchor_date - timedelta(days=anchor_date.weekday())
            return {"standard_date": monday.strftime("%Y-%m-%d"), "precision": "week", "confidence": 0.5}
        if "上周" in exp:
            monday = anchor_date - timedelta(days=anchor_date.weekday() + 7)
            return {"standard_date": monday.strftime("%Y-%m-%d"), "precision": "week", "confidence": 0.5}
        if "本月" in exp:
            return {"standard_date": anchor_date.strftime("%Y-%m") + "-01", "precision": "month", "confidence": 0.6}
        if "上月" in exp:
            y, m = anchor_date.year, anchor_date.month - 1
            if m == 0:
                y, m = y - 1, 12
            return {"standard_date": f"{y}-{m:02d}-01", "precision": "month", "confidence": 0.6}
        if "今年" in exp or "本年" in exp:
            return {"standard_date": f"{anchor_date.year}-01-01", "precision": "year", "confidence": 0.7}
        if "去年" in exp:
            return {"standard_date": f"{anchor_date.year - 1}-01-01", "precision": "year", "confidence": 0.7}
        if "上季度" in exp:
            q = (anchor_date.month - 1) // 3
            if q == 0:
                y, q = anchor_date.year - 1, 4
            else:
                y = anchor_date.year
            return {"standard_date": f"{y}-{(q - 1) * 3 + 1:02d}-01", "precision": "quarter", "confidence": 0.5}
        return None

    def _parse_json_response(self, content: str) -> list | dict | None:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
            if content.endswith("```"):
                content = content[:-3]
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _norm_event_times(self, results: list | dict | None) -> list[dict]:
        if isinstance(results, dict):
            return [results]
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict) and item.get("original")]
        return []

    def pick_best_event_time(self, event_times: list[dict]) -> tuple[str, str]:
        if not event_times:
            return "", ""
        best = max(event_times, key=lambda e: self._time_weight(e))
        return best.get("standard_date", ""), best.get("precision", "")

    def _time_weight(self, entry: dict) -> float:
        conf = entry.get("confidence", 0.5)
        prec = entry.get("precision", "")
        prec_weights = {"day": 1.0, "month": 0.9, "year": 0.6, "week": 0.7, "quarter": 0.5}
        prec_w = prec_weights.get(prec, 0.2)
        return conf * 1.0 + prec_w * 0.3