"""
知识图谱系统综合验收脚本
覆盖：服务健康 / API功能 / 数据完整性 / 智能特性 / 安全性 / 性能
"""
import asyncio
import json
import sys
import os
import time
import statistics
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

API_BASE = "http://localhost:8000/api"
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(TESTS_DIR, "graph_validation_report.json")


class Check:
    def __init__(self, phase, label, passed, detail="", duration_ms=0):
        self.phase = phase
        self.label = label
        self.passed = passed
        self.detail = detail
        self.duration_ms = duration_ms


class Validator:
    def __init__(self):
        self.checks: list[Check] = []
        self.errors: list[str] = []
        self.phase_results: dict[str, dict] = {}

    def check(self, phase, label, passed, detail="", duration_ms=0):
        c = Check(phase, label, passed, detail, duration_ms)
        self.checks.append(c)
        status = "PASS" if passed else "FAIL"
        dur = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        print(f"  [{status}] {label}{dur}")
        if detail:
            print(f"         {detail}")
        return c

    def phase_summary(self, phase):
        pc = [c for c in self.checks if c.phase == phase]
        passed = sum(1 for c in pc if c.passed)
        total = len(pc)
        score = passed / total if total > 0 else 0
        self.phase_results[phase] = {"passed": passed, "total": total, "score": score}
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"\n  {phase}: {bar} {score*100:.0f}% ({passed}/{total})\n")

    def total_score(self):
        if not self.checks:
            return 0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)


async def main():
    print("=" * 60)
    print("  知识图谱系统 综合验收")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = httpx.AsyncClient(timeout=120.0, base_url=API_BASE)
    v = Validator()

    try:
        # ================================================================
        # Phase 1: 服务健康检查
        # ================================================================
        phase = "1-服务健康"

        # 1.1 后端存活
        t0 = time.time()
        try:
            r = await client.get("/health")
            dur = (time.time() - t0) * 1000
            ok = r.status_code == 200
            v.check(phase, "后端服务存活", ok, f"HTTP {r.status_code}", dur)
        except Exception as e:
            v.check(phase, "后端服务存活", False, str(e))
            print("\n[中止] 后端不可用")
            return

        # 1.2 图谱统计
        t0 = time.time()
        r = await client.get("/graph/stats")
        dur = (time.time() - t0) * 1000
        if r.status_code == 200:
            stats = r.json()
            nodes = stats.get("node_count", 0)
            edges = stats.get("edge_count", 0)
            ok = nodes > 0 and edges > 0
            v.check(phase, "图谱数据非空", ok,
                    f"node_count={nodes}  edge_count={edges}  types={stats.get('node_types', {})}", dur)
        else:
            v.check(phase, "图谱数据非空", False, f"HTTP {r.status_code}")

        # 1.3 图谱探索
        t0 = time.time()
        r = await client.get("/graph/explore", params={"limit": 100, "hops": 1})
        dur = (time.time() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            nodes = data.get("nodes", []) or []
            edges = data.get("edges", []) or []
            ok = len(nodes) > 0 and len(edges) > 0
            v.check(phase, "图谱探索可用", ok,
                    f"nodes={len(nodes)} edges={len(edges)}", dur)
        else:
            v.check(phase, "图谱探索可用", False, f"HTTP {r.status_code}")

        v.phase_summary(phase)

        # ================================================================
        # Phase 2: 图数据模型验证
        # ================================================================
        phase = "2-数据模型"

        # 2.1 节点类型
        t0 = time.time()
        r = await client.get("/graph/stats")
        dur = (time.time() - t0) * 1000
        stats = r.json() if r.status_code == 200 else {}
        node_types = stats.get("node_types", {})
        expected_types = {"Entity", "Concept", "Event", "Document"}
        actual_types = set(node_types.keys())
        missing = expected_types - actual_types
        extra = actual_types - expected_types
        ok = len(missing) == 0
        detail = f"已有: {sorted(actual_types)}"
        if missing:
            detail += f"  缺失: {sorted(missing)}"
        if extra:
            detail += f"  额外: {sorted(extra)}"
        v.check(phase, f"核心节点类型({len(expected_types)}种)", ok, detail, dur)

        # 2.2 节点带 type 字段
        t0 = time.time()
        r = await client.get("/graph/explore", params={"limit": 100, "hops": 1})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        nodes = data.get("nodes", []) or []
        nodes_with_label = [n for n in nodes if n.get("id")]
        nodes_with_type = [n for n in nodes_with_label if n.get("type")]
        ok = len(nodes_with_type) > 0
        v.check(phase, "节点携带类型标签", ok,
                f"{len(nodes_with_type)}/{len(nodes_with_label)} 个有效节点有 type", dur)

        # 2.3 关系类型
        edges = data.get("edges", []) or []
        relation_types = set()
        edges_with_rel = 0
        edges_with_conf = 0
        for e in edges:
            rt = e.get("relation_type")
            if rt and rt != "RELATED_TO":
                relation_types.add(rt)
            if e.get("relation_type"):
                edges_with_rel += 1
            if e.get("confidence") is not None:
                edges_with_conf += 1
        ok = len(relation_types) >= 3
        v.check(phase, f"细分关系类型(>=3种)", ok,
                f"检测到 {len(relation_types)} 种: {sorted(relation_types)}", dur)

        # 2.4 边带 confidence
        ok = edges_with_conf > 0
        v.check(phase, "边携带置信度", ok,
                f"{edges_with_conf}/{len(edges)} 条边有 confidence")

        # 2.5 边带 relation_type
        ok = edges_with_rel > 0
        v.check(phase, "边携带 relation_type", ok,
                f"{edges_with_rel}/{len(edges)} 条边有 relation_type")

        v.phase_summary(phase)

        # ================================================================
        # Phase 3: 图谱 API 全量测试
        # ================================================================
        phase = "3-API功能"

        api_tests = [
            ("GET  /graph/stats", "/graph/stats", "GET", {}),
            ("GET  /graph/explore?entity=AI&hops=1", "/graph/explore", "GET",
             {"entity": "AI", "hops": 1, "limit": 50}),
            ("GET  /graph/explore?hops=2", "/graph/explore", "GET",
             {"entity": "AI", "hops": 2, "limit": 50}),
            ("GET  /graph/paths", "/graph/paths", "GET",
             {"source": "AI", "target": "机器学习", "max_hops": 3}),
            ("GET  /graph/conflicts", "/graph/conflicts", "GET",
             {"include_cycles": "true"}),
            ("POST /graph/communities", "/graph/communities", "POST",
             {},
             {"algorithm": "label_propagation"}),
            ("POST /graph/normalize", "/graph/normalize", "POST",
             {},
             {"entity_name": "AI"}),
            ("POST /graph/inference", "/graph/inference", "POST",
             {},
             {"rules": ["is_a_causes", "part_of_transitive"]}),
            ("POST /graph/fusion/scan", "/graph/fusion/scan", "POST",
             {},
             {"threshold": 0.95}),
        ]

        for label, path, method, params_dict, *rest in api_tests:
            t0 = time.time()
            try:
                if method == "GET":
                    params = params_dict
                    r = await client.get(path, params=params if isinstance(params, dict) else {})
                else:
                    body = rest[0] if rest else {}
                    r = await client.post(path, json=body)

                dur = (time.time() - t0) * 1000
                ok = r.status_code == 200
                detail = f"HTTP {r.status_code}"
                if ok and r.headers.get("content-type", "").startswith("application/json"):
                    try:
                        jd = r.json()
                        if isinstance(jd, dict):
                            keys = list(jd.keys())[:4]
                            detail += f"  keys={keys}"
                        elif isinstance(jd, list):
                            detail += f"  items={len(jd)}"
                    except Exception:
                        pass
                v.check(phase, label, ok, detail, dur)
            except Exception as e:
                v.check(phase, label, False, str(e))

        # 实体详情
        t0 = time.time()
        try:
            r = await client.get("/graph/entity/AI")
            dur = (time.time() - t0) * 1000
            ok = r.status_code == 200
            detail = f"HTTP {r.status_code}"
            if ok:
                entity = r.json()
                detail += f"  name={entity.get('name', '?')}  aliases={entity.get('aliases', [])}"
            v.check(phase, "GET  /graph/entity/{name}", ok, detail, dur)
        except Exception as e:
            v.check(phase, "GET  /graph/entity/{name}", False, str(e))

        v.phase_summary(phase)

        # ================================================================
        # Phase 4: 智能特性验证
        # ================================================================
        phase = "4-智能特性"

        # 4.1 实体归一化
        t0 = time.time()
        r = await client.post("/graph/normalize", json={"entity_name": "AI"})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        candidates = data.get("similar_entities", data.get("candidates", []))
        ok = isinstance(candidates, list) and len(candidates) > 0
        v.check(phase, "实体归一化有候选", ok,
                f"找到 {len(candidates) if isinstance(candidates, list) else 0} 个候选", dur)

        # 4.2 路径推理
        t0 = time.time()
        r = await client.get("/graph/paths", params={"source": "AI", "target": "机器学习", "max_hops": 4})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        paths = data.get("paths", [])
        ok = isinstance(paths, list)
        v.check(phase, "多跳路径推理", ok,
                f"找到 {len(paths) if isinstance(paths, list) else 0} 条路径", dur)

        # 4.3 矛盾检测
        t0 = time.time()
        r = await client.get("/graph/conflicts", params={"include_cycles": "true"})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        conflicts = data.get("conflicts", [])
        cycles = data.get("contradiction_cycles", data.get("cycles", []))
        ok = isinstance(conflicts, list) and isinstance(cycles, list)
        v.check(phase, "矛盾检测与环路", ok,
                f"conflicts={len(conflicts)}  cycles={len(cycles) if isinstance(cycles, list) else 0}", dur)

        # 4.4 社区检测
        t0 = time.time()
        r = await client.post("/graph/communities", json={"algorithm": "label_propagation"})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        communities = data.get("communities", [])
        ok = isinstance(communities, list)
        algo = data.get("algorithm", "?")
        v.check(phase, f"社区检测({algo})", ok,
                f"communities={len(communities) if isinstance(communities, list) else 0}", dur)

        # 4.5 推理规则
        t0 = time.time()
        r = await client.post("/graph/inference", json={"rules": ["is_a_causes", "part_of_transitive"]})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        results = data.get("results", data.get("inferred_relations", []))
        ok = isinstance(results, list)
        v.check(phase, "推理规则应用", ok,
                f"推断 {len(results) if isinstance(results, list) else 0} 条新关系", dur)

        # 4.6 知识融合扫描
        t0 = time.time()
        r = await client.post("/graph/fusion/scan", json={"threshold": 0.95})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        pairs = data.get("similar_pairs", data.get("candidates", []))
        ok = isinstance(pairs, list)
        v.check(phase, "知识融合扫描", ok,
                f"similar_pairs={len(pairs) if isinstance(pairs, list) else 0}", dur)

        v.phase_summary(phase)

        # ================================================================
        # Phase 5: 安全验证
        # ================================================================
        phase = "5-安全验证"

        # 5.1 阻止 DELETE
        t0 = time.time()
        r = await client.post("/graph/cypher", json={"query": "MATCH (n) DELETE n"})
        dur = (time.time() - t0) * 1000
        ok = r.status_code in (400, 403, 422)
        v.check(phase, "阻止 DELETE 操作", ok, f"HTTP {r.status_code}", dur)

        # 5.2 阻止 SET
        t0 = time.time()
        r = await client.post("/graph/cypher", json={"query": "MATCH (n) SET n.name='hack'"})
        dur = (time.time() - t0) * 1000
        ok = r.status_code in (400, 403, 422)
        v.check(phase, "阻止 SET 操作", ok, f"HTTP {r.status_code}", dur)

        # 5.3 允许 READ
        t0 = time.time()
        r = await client.post("/graph/cypher", json={"query": "MATCH (n) RETURN n LIMIT 5"})
        dur = (time.time() - t0) * 1000
        data = r.json() if r.status_code == 200 else {}
        rows = data.get("results", data.get("data", []))
        ok = r.status_code == 200 and isinstance(rows, list) and len(rows) > 0
        v.check(phase, "允许只读查询", ok,
                f"HTTP {r.status_code}  rows={len(rows) if isinstance(rows, list) else 0}", dur)

        # 5.4 密钥不暴露
        r = await client.get("/graph/stats")
        body = r.text.lower()
        ok = "api_key" not in body and "apikey" not in body and "secret" not in body
        v.check(phase, "API Key 不暴露", ok, "响应中无密钥关键字")

        v.phase_summary(phase)

        # ================================================================
        # Phase 6: 性能基准
        # ================================================================
        phase = "6-性能基准"

        perf_targets = [
            ("/graph/stats", 200),
            ("/graph/explore", 300),
            ("/graph/paths", 500),
            ("/graph/entity/AI", 200),
        ]

        for endpoint, max_ms in perf_targets:
            times = []
            for _ in range(3):
                t0 = time.time()
                params = {}
                if "entity" in endpoint:
                    endpoint_actual, _, entity = endpoint.rpartition("/")
                    endpoint_actual = endpoint_actual or endpoint
                    params = {"entity": entity, "hops": 1, "limit": 100}
                elif "paths" in endpoint:
                    params = {"source": "AI", "target": "机器学习", "max_hops": 3}
                else:
                    endpoint_actual = endpoint

                await client.get(f"/graph/{endpoint.split('/')[-1]}" if "/graph/" in endpoint else endpoint,
                                 params=params)
                times.append((time.time() - t0) * 1000)

            avg = statistics.mean(times)
            ok = avg < max_ms
            detail = f"avg={avg:.0f}ms  max={max(times):.0f}ms  limit={max_ms}ms"
            v.check(phase, f"{endpoint} < {max_ms}ms", ok, detail, avg)

        v.phase_summary(phase)

    except Exception as e:
        v.errors.append(f"异常: {e}")
        print(f"\n[异常] {e}")

    finally:
        await client.aclose()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  验收汇总")
    print("=" * 60)

    for phase_name in sorted(v.phase_results.keys()):
        pr = v.phase_results[phase_name]
        pct = pr["score"] * 100
        icon = "OK" if pr["score"] >= 0.9 else ("~ " if pr["score"] >= 0.6 else "!!")
        print(f"  [{icon}] {phase_name}: {pct:.0f}% ({pr['passed']}/{pr['total']})")

    total_passed = sum(1 for c in v.checks if c.passed)
    total_all = len(v.checks)
    overall = total_passed / total_all if total_all > 0 else 0

    bar = "=" * int(overall * 30)
    print(f"\n  总通过率: [{bar}>{'-' * (29 - int(overall * 30))}] {overall*100:.0f}% ({total_passed}/{total_all})")

    if overall >= 0.95:
        grade = "A+ (卓越)"
    elif overall >= 0.90:
        grade = "A  (优秀)"
    elif overall >= 0.80:
        grade = "B  (良好)"
    elif overall >= 0.60:
        grade = "C  (待改进)"
    else:
        grade = "D  (不合格)"

    print(f"  评级: {grade}")

    if v.errors:
        print(f"\n  [错误] {len(v.errors)} 个:")
        for e in v.errors:
            print(f"    - {e}")

    # Write report
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_pass_rate": round(overall, 3),
        "grade": grade,
        "phases": v.phase_results,
        "total_checks": total_all,
        "passed_checks": total_passed,
        "checks": [
            {
                "phase": c.phase, "label": c.label, "passed": c.passed,
                "detail": c.detail, "duration_ms": c.duration_ms,
            }
            for c in v.checks
        ],
        "errors": v.errors,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  报告: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())