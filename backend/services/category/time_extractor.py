import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

YEAR_PATTERN = re.compile(r"(\d{4})\s*年")
YEAR_MONTH_PATTERN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月")
DATE_PATTERN = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
QUARTER_PATTERN = re.compile(r"(\d{4})\s*年?\s*[Qq]([1-4])\s*(?:季度|财报)?")
RELATIVE_PATTERN = re.compile(r"(去年|今年|明年|上个月|下个月|上周|下周|昨天|今天|明天)")
SEASON_PATTERN = re.compile(r"(\d{4})\s*年\s*(春|夏|秋|冬)(?:季|天)?")

ERA_PATTERNS = [
    (re.compile(r"(公元前|BCE?)\s*(\d{1,4})\s*年"), lambda m: -(int(m.group(2)))),
    (re.compile(r"(\d{1,2})\s*世纪(?:\s*(\d{2})\s*年代)?"), lambda m: _century_to_year(m)),
]

SEASON_MONTH = {"春": 3, "夏": 6, "秋": 9, "冬": 12}
QUARTER_MONTH = {"1": 1, "2": 4, "3": 7, "4": 10}


def _century_to_year(match) -> int:
    century = int(match.group(1))
    decade = match.group(2)
    year = (century - 1) * 100
    if decade:
        year += int(decade)
    return year


def extract_event_times(text: str) -> list[dict]:
    """从文本中提取所有事件时间表达式"""
    events = []

    for m in YEAR_MONTH_PATTERN.finditer(text):
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            events.append({
                "type": "year_month",
                "year": year,
                "month": month,
                "label": f"{year}年{month}月",
                "span": m.span(),
                "text": m.group(),
            })

    for m in QUARTER_PATTERN.finditer(text):
        year = int(m.group(1))
        quarter = int(m.group(2))
        events.append({
            "type": "quarter",
            "year": year,
            "month": QUARTER_MONTH.get(str(quarter), 1),
            "quarter": quarter,
            "label": f"{year}年Q{quarter}",
            "span": m.span(),
            "text": m.group(),
        })

    for m in DATE_PATTERN.finditer(text):
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            events.append({
                "type": "date",
                "year": year,
                "month": month,
                "day": day,
                "label": f"{year}年{month}月{day}日",
                "span": m.span(),
                "text": m.group(),
            })

    for m in YEAR_PATTERN.finditer(text):
        year = int(m.group(1))
        already_covered = any(
            e["span"][0] <= m.start() < e["span"][1]
            for e in events
        )
        if 1000 <= year <= 2100 and not already_covered:
            events.append({
                "type": "year",
                "year": year,
                "label": f"{year}年",
                "span": m.span(),
                "text": m.group(),
            })

    for m in SEASON_PATTERN.finditer(text):
        year = int(m.group(1))
        season = m.group(2)
        events.append({
            "type": "season",
            "year": year,
            "month": SEASON_MONTH.get(season, 1),
            "season": season,
            "label": f"{year}年{season}季",
            "span": m.span(),
            "text": m.group(),
        })

    for pattern, converter in ERA_PATTERNS:
        for m in pattern.finditer(text):
            year = converter(m)
            events.append({
                "type": "era",
                "year": year,
                "label": m.group(),
                "span": m.span(),
                "text": m.group(),
            })

    events.sort(key=lambda e: (e.get("year", 0), e.get("month", 1), e.get("day", 1)))
    return events


def standardize_event_time(fact: str) -> Optional[str]:
    """标准化事件时间为可排序字符串 YYYY-MM"""
    events = extract_event_times(fact)
    if not events:
        return None
    primary = events[0]
    year = primary.get("year", 0)
    month = primary.get("month", 1)
    return f"{year:04d}-{month:02d}"


def is_historical(event_year: int, threshold_years: int = 5) -> bool:
    """判断事件时间是否已为历史"""
    current_year = datetime.now().year
    return (current_year - event_year) > threshold_years


def build_timeline_groups(
    knowledge_points: list[dict],
) -> dict[str, list[str]]:
    """构建时间线分组：按年月组织知识点"""
    timeline: dict[str, list[str]] = {}
    for kp in knowledge_points:
        fact = kp.get("fact", "")
        kp_id = kp.get("id", "")
        if not kp_id:
            continue
        event_time = standardize_event_time(fact)
        if event_time:
            key = event_time
        else:
            created = kp.get("created_at")
            if isinstance(created, str):
                created = created[:7]
            elif hasattr(created, "strftime"):
                created = created.strftime("%Y-%m")
            else:
                created = "unknown"
            key = created
        if key not in timeline:
            timeline[key] = []
        timeline[key].append(kp_id)
    return dict(sorted(timeline.items()))