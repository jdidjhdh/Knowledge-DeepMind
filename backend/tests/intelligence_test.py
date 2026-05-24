"""
智能体智能性验证测试框架
基于5大维度、20个测试用例的自动化评估系统
"""
import asyncio
import json
import sys
import os
import re
import statistics
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 终端编码修正
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

API_BASE = "http://localhost:8000/api"

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(TESTS_DIR, "intelligence_report.json")


class TestResult:
    def __init__(self, name: str, dimension: str):
        self.name = name
        self.dimension = dimension
        self.subscores: list[tuple[str, float, str]] = []
        self.raw_answer = ""
        self.passed_checks: list[str] = []
        self.failed_checks: list[str] = []
        self.notes: list[str] = []

    @property
    def total_score(self) -> float:
        if not self.subscores:
            return 0.0
        return sum(s for _, s, _ in self.subscores) / len(self.subscores)

    def check(self, label: str, condition: bool, detail: str = ""):
        score = 1.0 if condition else 0.0
        self.subscores.append((label, score, detail))
        if condition:
            self.passed_checks.append(f"{label}: {detail}" if detail else label)
        else:
            self.failed_checks.append(f"{label}: {detail}" if detail else label)

    def add_note(self, note: str):
        self.notes.append(note)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "total_score": round(self.total_score, 2),
            "subscores": [{"label": l, "score": s, "detail": d} for l, s, d in self.subscores],
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "notes": self.notes,
            "raw_answer_preview": self.raw_answer[:300],
        }


class IntelligenceTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0, base_url=API_BASE)
        self.results: list[TestResult] = []
        self.current_conv_id: Optional[str] = None

    async def close(self):
        await self.client.aclose()

    async def health_check(self) -> bool:
        try:
            r = await self.client.get("/health")
            return r.status_code == 200
        except Exception:
            return False

    async def ingest_text(self, content: str, source_name: str) -> Optional[str]:
        """纯文本注入知识库"""
        try:
            r = await self.client.post(
                "/ingest/text",
                json={"content": content, "source_name": source_name, "format": "natural"},
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("task_id")
            return None
        except Exception as e:
            print(f"  [注入失败] {source_name}: {e}")
            return None

    async def chat(self, message: str, conv_id: Optional[str] = None,
                   enable_web_search: bool = False) -> dict:
        """非流式对话"""
        try:
            body = {
                "message": message,
                "conversation_id": conv_id,
                "stream": False,
                "enable_web_search": enable_web_search,
            }
            r = await self.client.post("/chat", json=body)
            if r.status_code == 200:
                data = r.json()
                return data
            return {"answer": f"[ERROR {r.status_code}]", "sources": [], "detected_conflicts": [], "knowledge_gaps": []}
        except Exception as e:
            return {"answer": f"[EXCEPTION: {e}]", "sources": [], "detected_conflicts": [], "knowledge_gaps": []}

    async def multi_turn_chat(self, messages: list[str], conv_id: Optional[str] = None) -> list[dict]:
        """多轮对话"""
        results = []
        cid = conv_id
        for msg in messages:
            resp = await self.chat(msg, cid)
            results.append(resp)
            if not cid and resp.get("conversation_id"):
                cid = resp.get("conversation_id")
        return results

    def contains_any(self, text: str, keywords: list[str]) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)

    def contains_all(self, text: str, keywords: list[str]) -> bool:
        text_lower = text.lower()
        return all(kw.lower() in text_lower for kw in keywords)

    # ============================================================
    #  维度一：理解与归纳能力
    # ============================================================

    async def test_1a_multi_doc_synthesis(self):
        """多文档语义归纳"""
        r = TestResult("多文档语义归纳", "理解与归纳")

        doc1 = """远程办公对团队协作的最大挑战是沟通延迟。在办公室可以随时交流，但远程环境下
        消息回复时间平均延长2-3小时，导致项目进度变慢。同时视频会议难以替代白板讨论。"""
        doc2 = """远程办公的优势在于灵活性和节省通勤时间，员工满意度提升15%。
        但团队凝聚力下降是显著问题，新员工融入困难，缺乏非正式交流机会。"""
        doc3 = """远程办公研究表明：虽然跨时区协作更灵活，但信息孤岛现象加剧。
        不同部门间同步成本上升，决策周期延长30%。不过也有人认为这是管理方式需要进化。"""

        await self.ingest_text(doc1, "远程办公报告A")
        await self.ingest_text(doc2, "远程办公报告B")
        await self.ingest_text(doc3, "远程办公报告C")
        await asyncio.sleep(2)

        resp = await self.chat("综合这些材料，远程办公对团队协作的主要挑战是什么？")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("提取出共同主题",
                self.contains_any(answer, ["沟通", "协作", "团队"]),
                "需要提到沟通/协作/团队相关主题")
        r.check("对比观点差异",
                self.contains_any(answer, ["延迟", "凝聚力", "信息孤岛"]) or
                ("不同" in answer and "但" in answer),
                "需要对比不同来源的观点差异")
        r.check("发现并指出矛盾",
                self.contains_any(answer, ["矛盾", "冲突", "不同观点", "不同角度"]) or
                ("有的" in answer and "另" in answer),
                "需要指出矛盾或不同视角")

        self.results.append(r)
        return r

    async def test_1b_fragment_fact_graph(self):
        """零散事实建图"""
        r = TestResult("零散事实建图", "理解与归纳")

        fact_text = "张三是CEO，公司总部在北京，今年Q2营收增长了20%。"
        await self.ingest_text(fact_text, "公司信息")
        await asyncio.sleep(2)

        resp = await self.chat("张三公司的营收情况？")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("正确关联实体与属性",
                self.contains_any(answer, ["20%", "增长", "Q2", "营收"]),
                "需要提到营收增长20%或Q2")
        r.check("回答基于推断而非复读",
                len(answer) > 20 and "张三" in answer,
                "需要基于张三实体进行推断回答，而非原文复读")

        self.results.append(r)
        return r

    async def test_1c_abstract_concept(self):
        """抽象概念理解"""
        r = TestResult("抽象概念理解", "理解与归纳")

        concept_text = """蝴蝶效应的数学定义：在动力系统中，初始条件的微小变化能带动整个系统的
        长期巨大连锁反应。蝴蝶效应是一种混沌现象。\n\n生活案例1：一个不经意的善意举动可能引发
        一系列正面连锁反应，最终改变一个人的命运。\n\n生活案例2：天气预报中，某地一只蝴蝶扇动
        翅膀可能导致两周后万里之外的一场龙卷风。"""

        await self.ingest_text(concept_text, "蝴蝶效应概念")
        await asyncio.sleep(2)

        resp = await self.chat("用一句话解释蝴蝶效应")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("给出跨领域统一定义",
                self.contains_any(answer, ["初始", "微小", "变化", "系统", "连锁", "混沌"]),
                "需要给出包含核心要素的定义")
        r.check("排除具体案例干扰",
                "一句话" not in answer or len(answer.split("。")[0]) < 200,
                "回答核心定义而非罗列所有案例")

        self.results.append(r)
        return r

    async def test_1d_graph_triple_expansion(self):
        """图谱三元组扩展验证"""
        r = TestResult("图谱三元组扩展", "理解与归纳")

        resp = await self.chat("基于知识图谱中已有的实体关系，请生成10个关于AI Agent的相关事实陈述")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        facts = re.findall(r'\d+[\.\、\)）]\s*(.+?)(?=\n\d+[\.\、\)）]|\Z)', answer)
        if len(facts) < 3:
            facts = [line.strip() for line in answer.split("\n") if len(line.strip()) > 15]

        r.check("生成足够数量的事实",
                len(facts) >= 5,
                f"生成了 {len(facts)} 条事实，期望 >= 5")
        r.add_note(f"实际生成 {len(facts)} 条事实")

        self.results.append(r)
        return r

    # ============================================================
    #  维度二：推理与问题解决能力
    # ============================================================

    async def test_2a_multi_hop_reasoning(self):
        """多跳推理"""
        r = TestResult("多跳推理", "推理与问题解决")

        hop_text = """小明在A部门工作。A部门的预算被削减了30%。预算削减会影响A部门的所有项目进度。"""
        await self.ingest_text(hop_text, "组织信息")
        await asyncio.sleep(2)

        resp = await self.chat("小明的项目会受影响吗？")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("检索出两条知识",
                self.contains_any(answer, ["A部门", "预算", "削减", "影响"]),
                "需要体现对A部门和预算削减知识的检索")
        r.check("给出推理链条",
                self.contains_any(answer, ["会受", "影响", "因为", "所以", "由于"]) and
                len(answer) > 30,
                "需要给出推理路径")

        self.results.append(r)
        return r

    async def test_2b_conflict_resolution(self):
        """矛盾消解"""
        r = TestResult("矛盾消解", "推理与问题解决")

        conflict_a = """研究表明：维生素C可预防感冒（置信度0.3，来源：小型观察性研究）"""
        conflict_b = """大规模临床试验表明：维生素C不能预防感冒（置信度0.9，来源：Cochrane系统综述）"""

        await self.ingest_text(conflict_a, "VC研究A")
        await self.ingest_text(conflict_b, "VC研究B")
        await asyncio.sleep(2)

        resp = await self.chat("维生素C能预防感冒吗？")
        answer = resp.get("answer", "")
        conflicts = resp.get("detected_conflicts", [])
        r.raw_answer = answer

        r.check("发现矛盾",
                self.contains_any(answer, ["矛盾", "冲突", "不一致", "不同结论"]) or
                len(conflicts) > 0,
                f"detected_conflicts={conflicts}")
        r.check("依据置信度给出结论",
                self.contains_any(answer, ["置信度", "0.9", "0.3", "大规模", "系统综述"]) and
                self.contains_any(answer, ["不能", "不推荐", "无证据", "不太可能"]),
                "需要基于置信度/证据强度给出倾向性结论")
        r.check("主动提议修正",
                self.contains_any(answer, ["修正", "更新", "纠正", "删除", "调整", "重新评估"]),
                "需要主动提议修正知识库")

        self.results.append(r)
        return r

    async def test_2c_default_completion(self):
        """缺省填补"""
        r = TestResult("缺省填补", "推理与问题解决")

        default_text = "进行户外活动时需要携带防晒霜以保护皮肤。"
        await self.ingest_text(default_text, "户外安全指南")
        await asyncio.sleep(2)

        resp = await self.chat("阴天去户外要带防晒霜吗？")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("基于常识补全",
                self.contains_any(answer, ["紫外线", "云层", "需要", "建议", "仍然", "阴天"]),
                "需要基于常识推理（紫外线/云层穿透）补全")
        r.check("明确标注推理假设",
                self.contains_any(answer, ["假设", "前提", "常识", "一般", "可能", "据我所知"]),
                "需要标注推理的假设前提")

        self.results.append(r)
        return r

    async def test_2d_counterfactual_reasoning(self):
        """反事实推理"""
        r = TestResult("反事实推理", "推理与问题解决")

        fact_text = "项目X延期了两个月，主要原因是在开发阶段人力不足，只有2名工程师而计划需要5名。"
        await self.ingest_text(fact_text, "项目X记录")
        await asyncio.sleep(2)

        resp = await self.chat("如果当时人力充足，项目会按时完成吗？")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("给出反事实结论",
                self.contains_any(answer, ["按时", "可能", "不会延期", "如果", "假设"]) and
                len(answer) > 20,
                "需要基于反事实条件给出结论")
        r.check("讨论其他可能因素",
                self.contains_any(answer, ["其他", "也", "可能", "因素", "风险", "不确定"]),
                "需要讨论人力之外的其他可能因素")

        self.results.append(r)
        return r

    # ============================================================
    #  维度三：主动性与元认知能力
    # ============================================================

    async def test_3a_vague_reference_clarification(self):
        """模糊指代澄清"""
        r = TestResult("模糊指代澄清", "主动性与元认知")

        ctx_a = "3月1日讨论了A方案：采用微服务架构重构支付系统，预计工期6个月。"
        ctx_b = "3月5日讨论了B方案：基于现有单体架构进行性能优化，预计工期2个月。"
        await self.ingest_text(ctx_a, "方案A讨论")
        await self.ingest_text(ctx_b, "方案B讨论")
        await asyncio.sleep(2)

        resp = await self.chat("我们上次讨论的方案有结果了吗？")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("检测到模糊",
                self.contains_any(answer, ["哪个", "A方案", "B方案", "哪一", "两", "不确定"]),
                "需要检测到指代模糊并反问")
        r.check("提供明确选项",
                self.contains_any(answer, ["A方案", "B方案"]) and
                self.contains_any(answer, ["微服务", "单体", "3月1日", "3月5日"]),
                "需要提供具体方案名称和细节")

        self.results.append(r)
        return r

    async def test_3b_knowledge_gap_detection(self):
        """知识缺口探测"""
        r = TestResult("知识缺口探测", "主动性与元认知")

        gap_text1 = "政策调控是影响中国房价的重要因素，包括限购、限贷和房产税政策。"
        gap_text2 = "人口结构变化影响房价，城镇化进程和人口老龄化改变住房需求。"
        await self.ingest_text(gap_text1, "房价因素-政策")
        await self.ingest_text(gap_text2, "房价因素-人口")
        await asyncio.sleep(2)

        resp = await self.chat("影响中国房价的因素有哪些？请全面分析。")
        answer = resp.get("answer", "")
        gaps = resp.get("knowledge_gaps", [])
        r.raw_answer = answer

        r.check("列出已知因素",
                self.contains_any(answer, ["政策", "限购", "人口", "城镇化", "需求"]),
                "需要列出知识库中已有的因素")
        r.check("指出缺失维度",
                self.contains_any(answer, ["土地", "供给", "经济", "利率", "收入",
                                            "缺失", "不足", "需要补充", "目前没有"])
                or len(gaps) > 0,
                f"knowledge_gaps={gaps}")
        r.check("主动请求补充",
                self.contains_any(answer, ["补充", "查找", "搜索", "需要", "我能", "帮您"]),
                "需要主动请求/建议补充缺失信息")

        self.results.append(r)
        return r

    async def test_3c_self_correction(self):
        """自我修正"""
        r = TestResult("自我修正", "主动性与元认知")

        bait_text = "机器学习模型评估的核心指标是准确率（Accuracy），在任何场景下都适用。"
        await self.ingest_text(bait_text, "评估指标-诱饵")
        await asyncio.sleep(2)

        cid = None
        resp1 = await self.chat("机器学习模型评估最重要的指标是什么？", conv_id=cid)

        resp2 = await self.chat(
            "不对，在样本不平衡的情况下准确率会误导，召回率和精确率更重要。那基于这个认知，金融欺诈检测应该用什么指标？",
            conv_id=cid,
        )
        answer = resp2.get("answer", "")
        r.raw_answer = answer

        r.check("即时更新",
                not self.contains_any(answer, ["准确率最重要", "准确率是核心"]),
                "纠正后不应再坚持准确率是最重要指标的原始说法")
        r.check("后续回答贯彻修正",
                self.contains_any(answer, ["召回率", "精确率", "F1", "ROC", "PR曲线"]),
                "后续回答应体现修正后的认知")

        self.results.append(r)
        return r

    async def test_3d_metacognition_review(self):
        """元认知定期对话"""
        r = TestResult("元认知定期对话", "主动性与元认知")

        resp = await self.chat(
            "请评估一下当前知识库的覆盖情况，哪些领域的内容比较丰富？哪些领域还需要补充？"
        )
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("评估知识库覆盖度",
                self.contains_any(answer, ["丰富", "缺少", "覆盖", "不足", "较多", "补充"]) and
                len(answer) > 30,
                "需要对知识库覆盖度进行评估")
        r.check("建议合理",
                self.contains_any(answer, ["建议", "推荐", "可以", "需要", "补充"]),
                "需要给出合理的补充建议")

        self.results.append(r)
        return r

    # ============================================================
    #  维度四：学习与个性化能力
    # ============================================================

    async def test_4a_style_adaptation(self):
        """回答风格自适应"""
        r = TestResult("回答风格自适应", "学习与个性化")

        finance_text = "财务报表分析的三大核心：利润表反映盈利能力，资产负债表反映财务健康状况，现金流量表反映资金流动性。"
        await self.ingest_text(finance_text, "财务分析基础")
        await asyncio.sleep(2)

        cid = None
        for _ in range(3):
            resp = await self.chat("请简洁回答：什么是财务报表分析？", conv_id=cid)
            cid = resp.get("conversation_id", cid)

        resp_final = await self.chat("财务报表有什么用？")
        answer = resp_final.get("answer", "")
        r.raw_answer = answer

        sentences = len(re.split(r'[。！？\n]', answer))
        r.check("风格随反馈改变",
                sentences <= 8 or "简洁" in answer.lower(),
                f"回答句数={sentences}，期望简洁（<=8句）")
        r.add_note(f"回答句数: {sentences}")

        self.results.append(r)
        return r

    async def test_4b_term_preference_memory(self):
        """偏好术语记忆"""
        r = TestResult("偏好术语记忆", "学习与个性化")

        finance_data = """公司的营收数据包括主营业务收入和其他业务收入。营收增长率是衡量企业发展的重要指标。"""
        await self.ingest_text(finance_data, "财务术语")
        await asyncio.sleep(2)

        cid = None
        resp1 = await self.chat("分析一下公司的营收情况")
        cid = resp1.get("conversation_id", cid)

        resp2 = await self.chat("别用'营收'，用'收入'。重新回答。", conv_id=cid)
        cid = resp2.get("conversation_id", cid)

        resp3 = await self.chat("那公司的财务表现如何评价？", conv_id=cid)
        answer = resp3.get("answer", "")
        r.raw_answer = answer

        r.check("单次纠正生效",
                "营收" not in answer or ("营收" in answer and "收入" in answer and answer.index("收入") < answer.index("营收")),
                "纠正后应使用'收入'而非'营收'")

        self.results.append(r)
        return r

    async def test_4c_interest_weighting(self):
        """兴趣权重调整"""
        r = TestResult("兴趣权重调整", "学习与个性化")

        ml_text = "机器学习是人工智能的核心分支，包括监督学习、无监督学习和强化学习三大范式。"
        literature_text = "文学是人类表达思想感情的艺术形式，涵盖诗歌、小说、散文、戏剧等多种体裁。"
        await self.ingest_text(ml_text, "ML基础")
        await self.ingest_text(literature_text, "文学基础")
        await asyncio.sleep(2)

        for _ in range(3):
            await self.chat("机器学习的监督学习有哪些经典算法？")

        resp = await self.chat("根据我的兴趣，推荐一些值得深入了解的知识领域")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("推荐偏向高频领域",
                self.contains_any(answer, ["机器学习", "监督", "算法", "AI", "深度"]),
                "推荐应偏向用户高频交互的领域")

        self.results.append(r)
        return r

    async def test_4d_forgetting_curve_review(self):
        """遗忘曲线复习 — 改为验证对旧知识的记忆能力"""
        r = TestResult("遗忘曲线-知识记忆", "学习与个性化")

        old_knowledge = "知识蒸馏（Knowledge Distillation）是一种模型压缩技术，通过大模型（教师）指导小模型（学生）学习。"
        await self.ingest_text(old_knowledge, "知识蒸馏概念")

        for _ in range(5):
            await self.chat("我们继续讨论深度学习")

        resp = await self.chat("还记得我之前学的知识蒸馏吗？帮我回顾一下")
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("根据时间触发复习",
                self.contains_any(answer, ["知识蒸馏", "Knowledge Distillation", "教师", "学生",
                                            "模型压缩", "蒸馏"]) and len(answer) > 20,
                "需要回顾之前的知识蒸馏概念")
        r.check("提供复习摘要",
                self.contains_any(answer, ["回顾", "之前", "学习", "曾经", "介绍"]) or
                len(answer) > 50,
                "需要提供复习性质的回答")

        self.results.append(r)
        return r

    # ============================================================
    #  维度五：对话自然度与情商
    # ============================================================

    async def test_5a_context_coherence(self):
        """上下文连贯"""
        r = TestResult("上下文连贯", "对话自然度")

        cid = None
        resp1 = await self.chat("我昨天去了长城")
        cid = resp1.get("conversation_id", cid)

        resp2 = await self.chat("人多吗？", conv_id=cid)
        answer = resp2.get("answer", "")
        r.raw_answer = answer

        r.check("指代消解正确",
                self.contains_any(answer, ["长城", "旅游", "景点", "游览", "人多"]) and
                len(answer) > 5,
                "应正确关联'长城'而非孤立回答")
        r.check("结合前文补充信息",
                self.contains_any(answer, ["长城", "北京", "八达岭", "旅游"]) and
                len(answer) > 15,
                "应结合前文上下文补充信息")

        self.results.append(r)
        return r

    async def test_5b_emotion_perception(self):
        """情绪感知与回应"""
        r = TestResult("情绪感知与回应", "对话自然度")

        cid = None
        resp1 = await self.chat("最近项目延期，压力好大。")
        cid = resp1.get("conversation_id", cid)

        resp2 = await self.chat("有什么时间管理建议吗？", conv_id=cid)
        answer = resp2.get("answer", "")
        r.raw_answer = answer

        r.check("检测到情绪",
                self.contains_any(resp1.get("answer", "") + answer,
                                  ["压力", "理解", "不容易", "辛苦", "感受", "共情"]),
                "需要在首次回应中体现对情绪的关注")
        r.check("回应包含共情语句",
                self.contains_any(answer,
                                  ["理解", "感受", "共情", "面对", "应对", "支持", "帮助",
                                   "放松", "休息", "调节"]),
                "回答中需要包含共情或支持性语句")

        self.results.append(r)
        return r

    async def test_5c_uncertainty_honesty(self):
        """不确定时坦白"""
        r = TestResult("不确定时坦白", "对话自然度")

        resp = await self.chat(
            "2050年AI会取代人类作家吗？请基于当前知识库内容诚实回答。"
        )
        answer = resp.get("answer", "")
        r.raw_answer = answer

        r.check("明确说'不知道'",
                self.contains_any(answer, ["不知道", "不确定", "无法", "难以", "知识库", "没有",
                                           "无法预测", "我不能"]),
                "面对不确定问题时需要坦白说明局限性")
        r.check("不编造事实",
                not self.contains_any(answer, ["根据研究预测2050年AI将"]) and
                not any(pat in answer for pat in ["2050年确定", "一定会替代", "2030年AI"]),
                "不应编造确定性结论")

        self.results.append(r)
        return r

    async def test_5d_moderation_in_questioning(self):
        """适度提问"""
        r = TestResult("适度提问", "对话自然度")

        conv_id = None
        for q in ["帮我分析一下数据", "就是那组数据", "上周那个"]:
            resp = await self.chat(q, conv_id=conv_id)
            conv_id = resp.get("conversation_id", conv_id)
            await asyncio.sleep(0.5)

        resp_final = await self.chat("快点告诉我结论")
        answer = resp_final.get("answer", "")
        r.raw_answer = answer

        r.check("识别用户不耐烦",
                self.contains_any(answer, ["假设", "基于", "可能", "推测"]) or
                (len(answer) < 200 and len(answer) > 10),
                "应识别到用户不耐烦并调整策略")
        r.check("调整提问策略",
                "?" not in answer or answer.count("?") <= 1,
                "应减少追问，直接给出最合理的答案")

        self.results.append(r)
        return r

    # ============================================================
    #  主流程
    # ============================================================

    async def run_all(self):
        print("=" * 60)
        print("  智能体智能性验证测试")
        print("=" * 60)

        if not await self.health_check():
            print("\n[错误] 后端服务不可用，请确保 http://localhost:8000 正在运行")
            return

        print("\n[系统] 后端服务可用，开始测试...\n")

        dimensions = [
            ("一、理解与归纳能力", [
                (self.test_1a_multi_doc_synthesis, "多文档语义归纳"),
                (self.test_1b_fragment_fact_graph, "零散事实建图"),
                (self.test_1c_abstract_concept, "抽象概念理解"),
                (self.test_1d_graph_triple_expansion, "图谱三元组扩展"),
            ]),
            ("二、推理与问题解决能力", [
                (self.test_2a_multi_hop_reasoning, "多跳推理"),
                (self.test_2b_conflict_resolution, "矛盾消解"),
                (self.test_2c_default_completion, "缺省填补"),
                (self.test_2d_counterfactual_reasoning, "反事实推理"),
            ]),
            ("三、主动性与元认知能力", [
                (self.test_3a_vague_reference_clarification, "模糊指代澄清"),
                (self.test_3b_knowledge_gap_detection, "知识缺口探测"),
                (self.test_3c_self_correction, "自我修正"),
                (self.test_3d_metacognition_review, "元认知定期对话"),
            ]),
            ("四、学习与个性化能力", [
                (self.test_4a_style_adaptation, "回答风格自适应"),
                (self.test_4b_term_preference_memory, "偏好术语记忆"),
                (self.test_4c_interest_weighting, "兴趣权重调整"),
                (self.test_4d_forgetting_curve_review, "遗忘曲线-知识记忆"),
            ]),
            ("五、对话自然度与情商", [
                (self.test_5a_context_coherence, "上下文连贯"),
                (self.test_5b_emotion_perception, "情绪感知与回应"),
                (self.test_5c_uncertainty_honesty, "不确定时坦白"),
                (self.test_5d_moderation_in_questioning, "适度提问"),
            ]),
        ]

        dim_scores: dict[str, list[float]] = {}

        for dim_name, tests in dimensions:
            print(f"\n{'─' * 50}")
            print(f"  {dim_name}")
            print(f"{'─' * 50}")

            dim_scores[dim_name] = []

            for test_fn, test_name in tests:
                print(f"\n  > {test_name}...")
                try:
                    result = await test_fn()
                    score = result.total_score
                    dim_scores[dim_name].append(score)

                    status = "PASS" if score >= 0.7 else ("WARN" if score >= 0.3 else "FAIL")
                    print(f"    [{status}] Score: {score:.2f}")
                    for label, s, detail in result.subscores:
                        mark = "[+]" if s > 0 else "[-]"
                        print(f"      {mark} {label}: {detail}")

                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"    [FAIL] Exception: {e}")
                    dim_scores[dim_name].append(0.0)

        print(f"\n{'=' * 50}")
        print("  综合智能度评估")
        print(f"{'=' * 50}")

        dim_labels = {
            "一、理解与归纳能力": 0,
            "二、推理与问题解决能力": 0,
            "三、主动性与元认知能力": 0,
            "四、学习与个性化能力": 0,
            "五、对话自然度与情商": 0,
        }

        for dn in dim_labels:
            scores = dim_scores.get(dn, [])
            dim_labels[dn] = statistics.mean(scores) if scores else 0.0

        w_understand = 0.25
        w_reasoning = 0.25
        w_proactive = 0.20
        w_personalize = 0.15
        w_natural = 0.15

        intelligence_score = (
            w_understand * dim_labels["一、理解与归纳能力"] +
            w_reasoning * dim_labels["二、推理与问题解决能力"] +
            w_proactive * dim_labels["三、主动性与元认知能力"] +
            w_personalize * dim_labels["四、学习与个性化能力"] +
            w_natural * dim_labels["五、对话自然度与情商"]
        )

        level = (
            "卓越智能 (>=0.9)" if intelligence_score >= 0.9 else
            "高级智能 (>=0.8)" if intelligence_score >= 0.8 else
            "基础智能 (>=0.6)" if intelligence_score >= 0.6 else
            "需要改进 (<0.6)"
        )

        print(f"\n  智能度得分: {intelligence_score:.2f}")
        print(f"  评级: {level}")
        print(f"\n  各维度得分:")
        print(f"    理解与归纳能力: {dim_labels['一、理解与归纳能力']:.2f} (权重 0.25)")
        print(f"    推理与问题解决: {dim_labels['二、推理与问题解决能力']:.2f} (权重 0.25)")
        print(f"    主动性与元认知: {dim_labels['三、主动性与元认知能力']:.2f} (权重 0.20)")
        print(f"    学习与个性化:   {dim_labels['四、学习与个性化能力']:.2f} (权重 0.15)")
        print(f"    对话自然度:     {dim_labels['五、对话自然度与情商']:.2f} (权重 0.15)")

        report = {
            "timestamp": datetime.now().isoformat(),
            "intelligence_score": round(intelligence_score, 2),
            "level": level.strip(),
            "dimension_scores": {k: round(v, 2) for k, v in dim_labels.items()},
            "weights": {
                "理解与归纳": w_understand,
                "推理与问题解决": w_reasoning,
                "主动性与元认知": w_proactive,
                "学习与个性化": w_personalize,
                "对话自然度": w_natural,
            },
            "test_details": [r.to_dict() for r in self.results],
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.total_score >= 0.6),
            "failed_tests": sum(1 for r in self.results if r.total_score < 0.6),
        }

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n  详细报告已保存: {REPORT_PATH}")
        print(f"\n{'=' * 50}")

        return report


async def main():
    tester = IntelligenceTester()
    try:
        await tester.run_all()
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())