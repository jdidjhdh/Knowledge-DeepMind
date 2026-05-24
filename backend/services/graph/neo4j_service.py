import logging
import math
import re
from typing import Optional

from neo4j import AsyncGraphDatabase

from config import Settings
from models import (
    KnowledgeTriple, EntityType, RelationType, GraphNodeDetail,
    GraphEdgeDetail, MultiHopPath, CommunityResult, RuleInference,
)

logger = logging.getLogger(__name__)

KNOWN_RELATION_TYPES = {rt.value for rt in RelationType}


def _classify_relation_type(relation_label: str) -> str:
    label_lower = relation_label.lower()
    type_map = {
        "is a": "IS_A", "isa": "IS_A", "是一种": "IS_A",
        "part of": "PART_OF", "partof": "PART_OF", "组成部分": "PART_OF",
        "instance of": "INSTANCE_OF", "实例": "INSTANCE_OF",
        "causes": "CAUSES", "cause": "CAUSES", "导致": "CAUSES", "引起": "CAUSES",
        "depends on": "DEPENDS_ON", "dependson": "DEPENDS_ON", "依赖": "DEPENDS_ON",
        "indicates": "INDICATES", "表明": "INDICATES", "指示": "INDICATES",
        "belongs to": "BELONGS_TO", "属于": "BELONGS_TO",
        "occurs at": "OCCURS_AT", "发生在": "OCCURS_AT",
        "before": "BEFORE", "之前": "BEFORE",
        "after": "AFTER", "之后": "AFTER",
        "evidenced by": "EVIDENCED_BY", "证据": "EVIDENCED_BY",
        "confirmed by": "CONFIRMED_BY", "确认": "CONFIRMED_BY",
        "conflicts with": "CONFLICTS_WITH", "冲突": "CONFLICTS_WITH", "矛盾": "CONFLICTS_WITH",
        "endorsed by": "ENDORSED_BY", "支持": "ENDORSED_BY",
        "revised to": "REVISED_TO", "修订为": "REVISED_TO",
    }
    for key, val in type_map.items():
        if key in label_lower:
            return val
    return "RELATED_TO"


def _infer_node_type(name: str, associated_type: Optional[str] = None) -> str:
    if associated_type:
        return associated_type
    name_lower = name.lower()
    time_keywords = ["年", "月", "日", "事件", "会议", "战争", "革命"]
    concept_keywords = ["理论", "概念", "模型", "框架", "范式", "方法", "策略"]
    doc_keywords = ["论文", "报告", "文献", "文档", "研究", "pdf", "doi"]
    for kw in time_keywords:
        if kw in name_lower:
            return "Event"
    for kw in concept_keywords:
        if kw in name_lower:
            return "Concept"
    for kw in doc_keywords:
        if kw in name_lower:
            return "Document"
    return "Entity"


class GraphService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = None

    async def initialize(self):
        await self._try_connect()

    async def _try_connect(self):
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
            await self.driver.verify_connectivity()
            await self._create_constraints()
            logger.info("Neo4j 连接成功")
            return True
        except Exception as e:
            logger.warning(f"Neo4j 连接失败: {e}, 使用内存模式")
            self.driver = None
            return False

    async def ensure_connected(self):
        if self.driver:
            try:
                await self.driver.verify_connectivity()
                return True
            except Exception:
                logger.warning("Neo4j 已断开，尝试重连...")
                self.driver = None
        return await self._try_connect()

    async def close(self):
        if self.driver:
            await self.driver.close()

    async def _create_constraints(self):
        if not self.driver:
            return
        try:
            async with self.driver.session() as session:
                constraints = [
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
                    "CREATE CONSTRAINT knowledge_atom_id IF NOT EXISTS FOR (k:KnowledgeAtom) REQUIRE k.knowledge_id IS UNIQUE",
                    "CREATE CONSTRAINT function_fqn IF NOT EXISTS FOR (f:Function) REQUIRE f.fqn IS UNIQUE",
                    "CREATE CONSTRAINT class_fqn IF NOT EXISTS FOR (c:Class) REQUIRE c.fqn IS UNIQUE",
                    "CREATE CONSTRAINT file_name IF NOT EXISTS FOR (f:File) REQUIRE f.name IS UNIQUE",
                    "CREATE CONSTRAINT algorithm_name IF NOT EXISTS FOR (a:Algorithm) REQUIRE a.name IS UNIQUE",
                    "CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
                ]
                for c in constraints:
                    try:
                        await session.run(c)
                    except Exception:
                        pass
        except Exception:
            pass

    # ==================== 三元组写入（增强版） ====================

    async def create_triples(self, triples: list[KnowledgeTriple], source_chunk_id: Optional[str] = None):
        if not self.driver:
            logger.info("Neo4j 未连接，跳过多条三元组写入")
            return
        try:
            async with self.driver.session() as session:
                for triple in triples:
                    rel_type = triple.relation_type.value if triple.relation_type else _classify_relation_type(triple.relation)
                    chunk_id = triple.source_chunk_id or source_chunk_id
                    subj_type = _infer_node_type(triple.subject)
                    obj_type = _infer_node_type(triple.object)
                    await session.run(
                        f"""
                        MERGE (s:{subj_type} {{name: $subject}})
                        SET s.type = $subj_type, s.updated_at = timestamp()
                        MERGE (o:{obj_type} {{name: $object}})
                        SET o.type = $obj_type, o.updated_at = timestamp()
                        MERGE (s)-[r:RELATES {{relation: $relation, relation_type: $rel_type}}]->(o)
                        SET r.confidence = $confidence,
                            r.source_knowledge_id = CASE
                                WHEN r.source_knowledge_id IS NULL OR r.source_knowledge_id = '' THEN $source_kid
                                WHEN $source_kid = '' OR $source_kid IS NULL THEN r.source_knowledge_id
                                WHEN $source_kid IN split(r.source_knowledge_id, ',') THEN r.source_knowledge_id
                                ELSE r.source_knowledge_id + ',' + $source_kid
                            END,
                            r.source_chunk_id = $chunk_id,
                            r.created_at = timestamp()
                        """,
                        subject=triple.subject,
                        object=triple.object,
                        relation=triple.relation,
                        rel_type=rel_type,
                        confidence=triple.confidence,
                        source_kid=triple.source_knowledge_id or "",
                        chunk_id=chunk_id or "",
                        subj_type=subj_type,
                        obj_type=obj_type,
                    )
            logger.info(f"写入 {len(triples)} 条三元组到 Neo4j")
        except Exception as e:
            logger.error(f"Neo4j 写入失败: {e}")

    # ==================== 实体归一化 ====================

    async def normalize_entity(self, entity_name: str, entity_type: str = "Entity", force_merge: bool = False) -> dict:
        if not self.driver:
            return {"status": "unavailable", "entity": entity_name}
        try:
            name_lower = entity_name.lower()
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS $name_part
                       OR toLower(n.canonical_name) CONTAINS $name_part
                       OR $name_part IN n.aliases
                    RETURN n.name as name, labels(n) as labels, n.aliases as aliases,
                           n.canonical_name as canonical
                    LIMIT 20
                    """,
                    name_part=name_lower[:30],
                )
                candidates = []
                async for record in result:
                    candidates.append({
                        "name": record["name"],
                        "labels": record["labels"],
                        "aliases": record.get("aliases", []) or [],
                        "canonical": record.get("canonical"),
                    })

                best_match = None
                if candidates:
                    for c in candidates:
                        cname = (c["canonical"] or c["name"]).lower()
                        if cname == name_lower:
                            best_match = c
                            break
                        for alias in (c["aliases"] or []):
                            if alias.lower() == name_lower:
                                best_match = c
                                break
                        if best_match:
                            break
                    if not best_match and force_merge:
                        best_match = candidates[0]

                if best_match and best_match["name"] != entity_name:
                    canon = best_match["canonical"] or best_match["name"]
                    await session.run(
                        """
                        MERGE (canon {name: $canon})
                        SET canon.canonical_name = $canon,
                            canon.aliases = coalesce(canon.aliases, []) + $new_alias
                        MERGE (duplicate {name: $entity_name})
                        SET duplicate.canonical_name = $canon
                        MERGE (duplicate)-[r:SAME_AS]->(canon)
                        SET r.confidence = 0.9, r.created_at = timestamp()
                        """,
                        canon=canon,
                        entity_name=entity_name,
                        new_alias=entity_name,
                    )
                    return {
                        "status": "merged",
                        "entity": entity_name,
                        "canonical": canon,
                        "matched_with": best_match["name"],
                    }

                return {
                    "status": "unique" if not candidates else "similar_found",
                    "entity": entity_name,
                    "candidates": candidates,
                }
        except Exception as e:
            logger.warning(f"实体归一化失败: {e}")
            return {"status": "error", "entity": entity_name, "error": str(e)}

    async def get_entity_aliases(self, entity_name: str) -> list[str]:
        if not self.driver:
            return [entity_name]
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    "MATCH (n {name: $name}) RETURN n.aliases as aliases, n.canonical_name as canon",
                    name=entity_name,
                )
                record = await result.single()
                if record:
                    aliases = record.get("aliases") or []
                    canon = record.get("canon")
                    all_names = [entity_name]
                    if canon:
                        all_names.append(canon)
                    all_names.extend(aliases)
                    return list(set(all_names))
        except Exception:
            pass
        return [entity_name]

    # ==================== 证据链创建 ====================

    async def create_evidence_chain(self, knowledge_id: str, chunk_id: str, chunk_content: str, source_path: str):
        if not self.driver:
            return
        try:
            async with self.driver.session() as session:
                snippet = chunk_content[:200] if len(chunk_content) > 200 else chunk_content
                await session.run(
                    """
                    MERGE (d:Document {name: $source_path})
                    SET d.type = 'Document', d.updated_at = timestamp()
                    MERGE (chunk:Document {name: $chunk_id})
                    SET chunk.content = $snippet,
                        chunk.type = 'Document',
                        chunk.source_path = $source_path
                    MERGE (k:KnowledgeAtom {knowledge_id: $knowledge_id})
                    MERGE (k)-[r:EVIDENCED_BY]->(chunk)
                    SET r.created_at = timestamp()
                    MERGE (chunk)-[r2:PART_OF]->(d)
                    SET r2.created_at = timestamp()
                    """,
                    knowledge_id=knowledge_id,
                    chunk_id=chunk_id,
                    snippet=snippet,
                    source_path=source_path,
                )
        except Exception as e:
            logger.warning(f"证据链创建失败: {e}")

    # ==================== 代码知识图谱构建 ====================

    async def create_code_structure(
        self, file_path: str, filename: str, language: str,
        static_result: dict, semantic: dict,
    ):
        if not self.driver:
            return

        functions = static_result.get("functions", [])
        classes = static_result.get("classes", [])
        imports = static_result.get("imports", [])
        complexity = static_result.get("complexity", [])
        call_graph = static_result.get("call_graph", {})
        algorithm_name = semantic.get("algorithm")
        concepts = semantic.get("related_concepts", [])
        summary = semantic.get("summary", "")

        try:
            async with self.driver.session() as session:
                file_fqn = f"{filename} [{language}]"
                await session.run(
                    """
                    MERGE (f:File {name: $file_path})
                    SET f.filename = $filename,
                        f.language = $language,
                        f.loc = $loc,
                        f.function_count = $func_count,
                        f.class_count = $class_count,
                        f.import_count = $import_count,
                        f.summary = $summary,
                        f.updated_at = timestamp()
                    """,
                    file_path=file_path, filename=filename,
                    language=language, summary=summary[:500],
                    loc=static_result.get("loc", 0),
                    func_count=len(functions),
                    class_count=len(classes),
                    import_count=len(imports),
                )

                # 创建函数节点
                for func in functions:
                    func_name = func.get("name", "")
                    func_fqn = f"{filename}::{func_name}"
                    comp = next((c for c in complexity if c.get("name") == func_name), {})
                    await session.run(
                        """
                        MERGE (fn:Function {fqn: $fqn})
                        SET fn.name = $name,
                            fn.file = $file_path,
                            fn.lineno = $lineno,
                            fn.line_count = $line_count,
                            fn.arg_count = $arg_count,
                            fn.args = $args,
                            fn.docstring = $docstring,
                            fn.has_loop = $has_loop,
                            fn.has_condition = $has_condition,
                            fn.has_try = $has_try,
                            fn.is_async = $is_async,
                            fn.complexity = $complexity,
                            fn.updated_at = timestamp()
                        MERGE (f:File {name: $file_path})
                        MERGE (f)-[r:DEFINES_FUNCTION {type: 'DEFINES_FUNCTION'}]->(fn)
                        SET r.created_at = timestamp()
                        """,
                        fqn=func_fqn, name=func_name, file_path=file_path,
                        lineno=func.get("lineno", 0),
                        line_count=func.get("line_count", 0),
                        arg_count=func.get("arg_count", 0),
                        args=func.get("args", []),
                        docstring=(func.get("docstring") or "")[:500],
                        has_loop=func.get("has_loop", False),
                        has_condition=func.get("has_condition", False),
                        has_try=func.get("has_try", False),
                        is_async=func.get("async", False),
                        complexity=comp.get("complexity", 0),
                    )

                # 创建类节点
                for cls in classes:
                    cls_name = cls.get("name", "")
                    cls_fqn = f"{filename}::{cls_name}"
                    methods = cls.get("methods", [])
                    await session.run(
                        """
                        MERGE (c:Class {fqn: $fqn})
                        SET c.name = $name,
                            c.file = $file_path,
                            c.lineno = $lineno,
                            c.line_count = $line_count,
                            c.bases = $bases,
                            c.methods = $methods,
                            c.method_count = $method_count,
                            c.docstring = $docstring,
                            c.updated_at = timestamp()
                        MERGE (f:File {name: $file_path})
                        MERGE (f)-[r:DEFINES_CLASS {type: 'DEFINES_CLASS'}]->(c)
                        SET r.created_at = timestamp()
                        """,
                        fqn=cls_fqn, name=cls_name, file_path=file_path,
                        lineno=cls.get("lineno", 0),
                        line_count=cls.get("line_count", 0),
                        bases=cls.get("bases", []),
                        methods=methods,
                        method_count=len(methods),
                        docstring=(cls.get("docstring") or "")[:500],
                    )
                    # 类方法关系
                    for m_name in methods:
                        m_fqn = f"{filename}::{m_name}"
                        await session.run(
                            """
                            MATCH (c:Class {fqn: $cls_fqn})
                            MATCH (fn:Function {fqn: $m_fqn})
                            MERGE (c)-[r:DEFINES_FUNCTION {type: 'DEFINES_FUNCTION'}]->(fn)
                            SET r.created_at = timestamp()
                            """,
                            cls_fqn=cls_fqn, m_fqn=m_fqn,
                        )
                    # 类继承关系
                    for base in cls.get("bases", []):
                        await session.run(
                            """
                            MERGE (c:Class {fqn: $cls_fqn})
                            MERGE (parent:Class {fqn: $base_fqn})
                            ON CREATE SET parent.name = $base_name
                            MERGE (c)-[r:RELATES {relation_type: 'IS_A', relation: 'inherits'}]->(parent)
                            SET r.confidence = 0.9, r.created_at = timestamp()
                            """,
                            cls_fqn=cls_fqn,
                            base_fqn=f"{filename}::{base}",
                            base_name=base,
                        )

                # 创建调用关系 (CALLS)
                for func in functions:
                    func_name = func.get("name", "")
                    func_fqn = f"{filename}::{func_name}"
                    for called in func.get("called_functions", [])[:20]:
                        called_fqn = f"{filename}::{called}"
                        await session.run(
                            """
                            MATCH (fn:Function {fqn: $func_fqn})
                            MERGE (target:Function {fqn: $called_fqn})
                            ON CREATE SET target.name = $called
                            MERGE (fn)-[r:CALLS {type: 'CALLS'}]->(target)
                            SET r.created_at = timestamp()
                            """,
                            func_fqn=func_fqn, called_fqn=called_fqn, called=called,
                        )

                # 创建导入依赖 (IMPORTS)
                for imp in imports[:30]:
                    module_name = imp.get("module", "")
                    if module_name:
                        await session.run(
                            """
                            MATCH (f:File {name: $file_path})
                            MERGE (m:Module {name: $module_name})
                            SET m.updated_at = timestamp()
                            MERGE (f)-[r:IMPORTS {type: 'IMPORTS'}]->(m)
                            SET r.created_at = timestamp()
                            """,
                            file_path=file_path, module_name=module_name,
                        )

                # 创建算法节点
                if algorithm_name:
                    await session.run(
                        """
                        MERGE (a:Algorithm {name: $algo_name})
                        SET a.updated_at = timestamp()
                        """,
                        algo_name=algorithm_name,
                    )
                    # 关联函数到算法
                    for func in functions:
                        func_fqn = f"{filename}::{func.get('name', '')}"
                        await session.run(
                            """
                            MATCH (fn:Function {fqn: $func_fqn})
                            MATCH (a:Algorithm {name: $algo_name})
                            MERGE (fn)-[r:IMPLEMENTS_ALGORITHM {type: 'IMPLEMENTS_ALGORITHM'}]->(a)
                            SET r.created_at = timestamp()
                            """,
                            func_fqn=func_fqn, algo_name=algorithm_name,
                        )

                # 创建技术概念节点
                for concept in concepts:
                    await session.run(
                        """
                        MERGE (c:Concept {name: $concept_name})
                        SET c.updated_at = timestamp()
                        """,
                        concept_name=concept,
                    )
                    for func in functions:
                        func_fqn = f"{filename}::{func.get('name', '')}"
                        await session.run(
                            """
                            MATCH (fn:Function {fqn: $func_fqn})
                            MATCH (c:Concept {name: $concept_name})
                            MERGE (fn)-[r:RELATED_TO_CONCEPT {type: 'RELATED_TO_CONCEPT'}]->(c)
                            SET r.created_at = timestamp()
                            """,
                            func_fqn=func_fqn, concept_name=concept,
                        )

            logger.info(
                f"[代码图谱] 存入: {filename} ({language}), "
                f"{len(functions)}函数, {len(classes)}类, {len(imports)}导入, "
                f"算法={algorithm_name}, 概念={len(concepts)}"
            )
        except Exception as e:
            logger.error(f"[代码图谱] 存储失败: {filename} - {e}")

    # ==================== 代码图谱查询 ====================

    async def get_file_code_structure(self, file_path: str) -> dict:
        if not self.driver:
            return {"file": file_path, "functions": [], "classes": [], "imports": []}
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (f:File)
                    WHERE f.name = $file_path OR f.name ENDS WITH $file_path
                    OPTIONAL MATCH (f)-[:DEFINES_FUNCTION]->(fn:Function)
                    OPTIONAL MATCH (f)-[:DEFINES_CLASS]->(c:Class)
                    OPTIONAL MATCH (f)-[:IMPORTS]->(m:Module)
                    OPTIONAL MATCH (fn)-[r:CALLS]->(called:Function)
                    RETURN f.name as file_name, f.language as language, f.summary as summary,
                           f.function_count as func_count, f.class_count as cls_count,
                           collect(DISTINCT {name: fn.name, fqn: fn.fqn, lineno: fn.lineno,
                                   args: fn.args, complexity: fn.complexity,
                                   docstring: fn.docstring}) as functions,
                           collect(DISTINCT {name: c.name, fqn: c.fqn, bases: c.bases,
                                   methods: c.methods}) as classes,
                           collect(DISTINCT {name: m.name}) as modules,
                           collect(DISTINCT {from: fn.name, to: called.name}) as calls
                    """,
                    file_path=file_path,
                )
                record = await result.single()
                if not record:
                    return {"file": file_path, "functions": [], "classes": [], "imports": []}

                funcs = [f for f in (record.get("functions") or []) if f.get("name")]
                clses = [c for c in (record.get("classes") or []) if c.get("name")]
                mods = [m for m in (record.get("modules") or []) if m.get("name")]
                calls = [c for c in (record.get("calls") or []) if c.get("from") and c.get("to")]

                return {
                    "file": file_path,
                    "language": record.get("language"),
                    "summary": record.get("summary"),
                    "function_count": record.get("func_count", 0),
                    "class_count": record.get("cls_count", 0),
                    "functions": funcs,
                    "classes": clses,
                    "modules": mods,
                    "calls": calls,
                }
        except Exception as e:
            logger.warning(f"代码结构查询失败: {e}")
            return {"file": file_path, "functions": [], "classes": [], "imports": []}

    async def find_functions_by_name(self, function_name: str, limit: int = 20) -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (fn:Function)
                    WHERE toLower(fn.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (fn)-[:IMPLEMENTS_ALGORITHM]->(a:Algorithm)
                    OPTIONAL MATCH (fn)-[:CALLS]->(called:Function)
                    OPTIONAL MATCH (caller:Function)-[:CALLS]->(fn)
                    RETURN fn.name as name, fn.fqn as fqn, fn.file as file,
                           fn.lineno as lineno, fn.args as args,
                           fn.complexity as complexity, fn.docstring as docstring,
                           a.name as algorithm,
                           collect(DISTINCT called.name) as calls_out,
                           collect(DISTINCT caller.name) as calls_in
                    LIMIT $limit
                    """,
                    name=function_name, limit=limit,
                )
                records = await result.data()
                return [
                    {
                        "name": r.get("name"),
                        "fqn": r.get("fqn"),
                        "file": r.get("file"),
                        "lineno": r.get("lineno"),
                        "args": r.get("args"),
                        "complexity": r.get("complexity"),
                        "docstring": (r.get("docstring") or "")[:200],
                        "algorithm": r.get("algorithm"),
                        "called_functions": [c for c in (r.get("calls_out") or []) if c],
                        "callers": [c for c in (r.get("calls_in") or []) if c],
                    }
                    for r in records if r.get("name")
                ]
        except Exception as e:
            logger.warning(f"按名称查询函数失败: {e}")
            return []

    async def find_functions_by_algorithm(self, algorithm_name: str) -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (fn:Function)-[:IMPLEMENTS_ALGORITHM]->(a:Algorithm)
                    WHERE toLower(a.name) CONTAINS toLower($algo)
                    RETURN fn.name as name, fn.fqn as fqn, fn.file as file,
                           fn.lineno as lineno, fn.complexity as complexity,
                           fn.args as args, fn.docstring as docstring,
                           a.name as algorithm
                    LIMIT 30
                    """,
                    algo=algorithm_name,
                )
                records = await result.data()
                return [
                    {
                        "name": r.get("name"), "fqn": r.get("fqn"),
                        "file": r.get("file"), "lineno": r.get("lineno"),
                        "complexity": r.get("complexity"),
                        "args": r.get("args"),
                        "docstring": (r.get("docstring") or "")[:200],
                        "algorithm": r.get("algorithm"),
                    }
                    for r in records if r.get("name")
                ]
        except Exception as e:
            logger.warning(f"按算法查询函数失败: {e}")
            return []

    async def find_functions_by_concept(self, concept_name: str) -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (fn:Function)-[:RELATED_TO_CONCEPT]->(c:Concept)
                    WHERE toLower(c.name) CONTAINS toLower($concept)
                    OPTIONAL MATCH (fn)-[:IMPLEMENTS_ALGORITHM]->(a:Algorithm)
                    RETURN fn.name as name, fn.fqn as fqn, fn.file as file,
                           fn.lineno as lineno, fn.complexity as complexity,
                           c.name as concept, a.name as algorithm
                    LIMIT 30
                    """,
                    concept=concept_name,
                )
                records = await result.data()
                return [
                    {
                        "name": r.get("name"), "fqn": r.get("fqn"),
                        "file": r.get("file"), "lineno": r.get("lineno"),
                        "complexity": r.get("complexity"),
                        "concept": r.get("concept"),
                        "algorithm": r.get("algorithm"),
                    }
                    for r in records if r.get("name")
                ]
        except Exception as e:
            logger.warning(f"按概念查询函数失败: {e}")
            return []

    async def get_code_call_graph(self, file_path: str = "", limit: int = 100) -> dict:
        if not self.driver:
            return {"nodes": [], "edges": []}
        try:
            async with self.driver.session() as session:
                if file_path:
                    result = await session.run(
                        """
                        MATCH (f:File {name: $file_path})-[:DEFINES_FUNCTION]->(fn:Function)
                        OPTIONAL MATCH (fn)-[r:CALLS]->(called:Function)
                        RETURN fn.name as source, called.name as target,
                               fn.file as source_file, called.file as target_file
                        LIMIT $limit
                        """,
                        file_path=file_path, limit=limit,
                    )
                else:
                    result = await session.run(
                        """
                        MATCH (fn:Function)-[r:CALLS]->(called:Function)
                        RETURN fn.name as source, called.name as target,
                               fn.file as source_file, called.file as target_file
                        LIMIT $limit
                        """
                    )

                records = await result.data()
                nodes_set = set()
                edges = []
                for r in records:
                    src = r.get("source")
                    tgt = r.get("target")
                    if src and tgt:
                        nodes_set.add(src)
                        nodes_set.add(tgt)
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "source_file": r.get("source_file", ""),
                            "target_file": r.get("target_file", ""),
                        })

                return {
                    "nodes": [{"name": n} for n in nodes_set],
                    "edges": edges,
                    "node_count": len(nodes_set),
                    "edge_count": len(edges),
                }
        except Exception as e:
            logger.warning(f"调用图查询失败: {e}")
            return {"nodes": [], "edges": []}

    async def list_algorithms(self) -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (a:Algorithm)<-[:IMPLEMENTS_ALGORITHM]-(fn:Function)
                    RETURN a.name as name, count(fn) as function_count,
                           collect(DISTINCT fn.name)[..10] as example_functions
                    ORDER BY function_count DESC
                    LIMIT 30
                    """
                )
                records = await result.data()
                return [
                    {
                        "name": r.get("name"),
                        "function_count": r.get("function_count", 0),
                        "example_functions": r.get("example_functions", []),
                    }
                    for r in records if r.get("name")
                ]
        except Exception as e:
            logger.warning(f"算法列表查询失败: {e}")
            return []

    async def list_code_modules(self) -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (m:Module)<-[:IMPORTS]-(f:File)
                    RETURN m.name as name, collect(DISTINCT f.filename)[..5] as used_by,
                           count(f) as file_count
                    ORDER BY file_count DESC
                    LIMIT 30
                    """
                )
                records = await result.data()
                return [
                    {
                        "name": r.get("name"),
                        "file_count": r.get("file_count", 0),
                        "used_by": r.get("used_by", []),
                    }
                    for r in records if r.get("name")
                ]
        except Exception as e:
            logger.warning(f"模块列表查询失败: {e}")
            return []

    # ==================== 图谱探索（增强版） ====================

    async def explore(self, entity: str = "", limit: int = 50, hops: int = 1) -> dict:
        if not self.driver:
            return {"nodes": [], "edges": [], "message": "Neo4j 未连接"}

        try:
            async with self.driver.session() as session:
                if entity:
                    hops_pattern = f"-[*1..{hops}]-" if hops > 1 else "-[r]-"
                    result = await session.run(
                        f"""
                        MATCH (s {{name: $entity}}){hops_pattern}(o)
                        RETURN s.name as source, labels(s) as source_labels,
                               type(r) as relation, r.relation as label,
                               o.name as target, labels(o) as target_labels,
                               r.confidence as confidence, r.relation_type as rel_type,
                               r.source_knowledge_id as source_kid
                        LIMIT $limit
                        """,
                        entity=entity, limit=limit,
                    )
                else:
                    result = await session.run(
                        """
                        MATCH (s)-[r]->(o)
                        RETURN s.name as source, labels(s) as source_labels,
                               type(r) as relation, r.relation as label,
                               o.name as target, labels(o) as target_labels,
                               r.confidence as confidence, r.relation_type as rel_type,
                               r.source_knowledge_id as source_kid
                        LIMIT $limit
                        """,
                        limit=limit,
                    )

                records = await result.data()
                nodes_map = {}
                edges = []
                for rec in records:
                    src = rec["source"]
                    tgt = rec["target"]
                    if not src or not tgt:
                        continue
                    if src not in nodes_map:
                        s_labels = rec.get("source_labels") or []
                        nodes_map[src] = {
                            "id": src, "label": src,
                            "type": s_labels[0] if s_labels else "Entity",
                            "degree": 0,
                        }
                    if tgt not in nodes_map:
                        t_labels = rec.get("target_labels") or []
                        nodes_map[tgt] = {
                            "id": tgt, "label": tgt,
                            "type": t_labels[0] if t_labels else "Entity",
                            "degree": 0,
                        }
                    nodes_map[src]["degree"] = nodes_map[src].get("degree", 0) + 1
                    nodes_map[tgt]["degree"] = nodes_map[tgt].get("degree", 0) + 1
                    relation_type = rec.get("rel_type") or rec.get("relation") or "RELATED_TO"
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "relation": rec.get("label") or rec.get("relation") or "",
                        "relation_type": relation_type,
                        "confidence": rec.get("confidence", 0.5),
                        "source_knowledge_id": rec.get("source_kid", ""),
                    })

                return {
                    "nodes": list(nodes_map.values()),
                    "edges": edges,
                    "node_count": len(nodes_map),
                    "edge_count": len(edges),
                }

        except Exception as e:
            logger.error(f"Neo4j 查询失败: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}

    # ==================== 实体邻居 ====================

    async def get_entity_neighbors(self, entity_name: str) -> dict:
        if not self.driver:
            return {"entity": entity_name, "neighbors": [], "avg_confidence": 0.5}
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e {name: $name})-[r]-(n)
                    RETURN n.name as neighbor, type(r) as rel_type,
                           r.relation as rel_label, r.confidence as confidence,
                           r.relation_type as graph_rel_type,
                           labels(n) as node_labels
                    """,
                    name=entity_name,
                )
                neighbors = []
                confidences = []
                async for record in result:
                    neighbors.append({
                        "name": record["neighbor"],
                        "relation": record.get("rel_label", record.get("rel_type", "")),
                        "relation_type": record.get("graph_rel_type", "RELATED_TO"),
                        "confidence": record.get("confidence", 0.5),
                        "node_type": (record.get("node_labels") or [None])[0],
                    })
                    confidences.append(record.get("confidence", 0.5))
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
                return {"entity": entity_name, "neighbors": neighbors, "avg_confidence": avg_conf}
        except Exception as e:
            logger.warning(f"获取邻居失败: {e}")
            return {"entity": entity_name, "neighbors": [], "avg_confidence": 0.5}

    # ==================== 多跳路径推理 ====================

    async def find_paths(self, source: str, target: str, max_hops: int = 4) -> MultiHopPath:
        if not self.driver:
            return MultiHopPath(paths=[], relations=[], path_weights=[], source_entity=source, target_entity=target, max_hops=max_hops)
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    f"""
                    MATCH path = (s {{name: $source}})-[*1..{max_hops}]-(t {{name: $target}})
                    RETURN nodes(path) as nodes, relationships(path) as rels
                    LIMIT 20
                    """,
                    source=source, target=target,
                )
                paths = []
                relations_list = []
                weights = []
                async for record in result:
                    node_names = [n.get("name", str(n.id)) for n in record["nodes"]]
                    rel_names = [r.get("relation", type(r).__name__) for r in record["rels"]]
                    conf_sum = sum(r.get("confidence", 0.5) for r in record["rels"])
                    paths.append(node_names)
                    relations_list.append(rel_names)
                    weights.append(round(conf_sum / max(len(record["rels"]), 1), 3))
                return MultiHopPath(
                    paths=paths, relations=relations_list, path_weights=weights,
                    source_entity=source, target_entity=target, max_hops=max_hops,
                )
        except Exception as e:
            logger.warning(f"多跳路径查询失败: {e}")
            return MultiHopPath(paths=[], relations=[], path_weights=[], source_entity=source, target_entity=target, max_hops=max_hops)

    # ==================== 矛盾检测 ====================

    async def detect_conflicts(self, fact: str = "", entity: str = "") -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                if fact:
                    result = await session.run(
                        """
                        MATCH (a)-[r:CONFLICTS_WITH]-(b)
                        WHERE r.description CONTAINS $fact_part
                        RETURN a.name as entity_a, b.name as entity_b,
                               r.description as description, r.confidence as confidence
                        LIMIT 20
                        """,
                        fact_part=fact[:50],
                    )
                elif entity:
                    result = await session.run(
                        """
                        MATCH (e {name: $entity})-[r:CONFLICTS_WITH]-(n)
                        RETURN e.name as entity_a, n.name as entity_b,
                               coalesce(r.description, r.relation, '未知冲突') as description,
                               r.confidence as confidence
                        LIMIT 20
                        """,
                        entity=entity,
                    )
                else:
                    result = await session.run(
                        """
                        MATCH (a)-[r:RELATES]->(b)
                        WHERE r.relation_type = 'CONFLICTS_WITH'
                        RETURN a.name as entity_a, b.name as entity_b,
                               r.relation as description, r.confidence as confidence
                        LIMIT 20
                        """
                    )

                conflicts = []
                async for record in result:
                    conflicts.append({
                        "entity_a": record["entity_a"],
                        "entity_b": record["entity_b"],
                        "description": record.get("description", ""),
                        "confidence": record.get("confidence", 0.5),
                    })

                if entity and not conflicts:
                    result2 = await session.run(
                        """
                        MATCH (e {name: $entity})-[r]-(n)
                        WHERE r.relation_type = 'CONFLICTS_WITH'
                        RETURN e.name as entity_a, n.name as entity_b,
                               r.relation as description, r.confidence as confidence
                        LIMIT 20
                        """,
                        entity=entity,
                    )
                    async for record in result2:
                        conflicts.append({
                            "entity_a": record["entity_a"],
                            "entity_b": record["entity_b"],
                            "description": record.get("description", ""),
                            "confidence": record.get("confidence", 0.5),
                        })

                return conflicts
        except Exception as e:
            logger.warning(f"矛盾检测失败: {e}")
            return []

    async def detect_contradiction_cycles(self, max_depth: int = 5) -> list[list[str]]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    f"""
                    MATCH path = (a)-[r:RELATES*1..{max_depth}]->(a)
                    WHERE all(rel IN relationships(path) WHERE rel.relation_type = 'CONFLICTS_WITH')
                    RETURN [n IN nodes(path) | n.name] as cycle_nodes
                    LIMIT 20
                    """
                )
                cycles = []
                async for record in result:
                    cycles.append(record["cycle_nodes"])
                return cycles
        except Exception as e:
            logger.warning(f"矛盾环路检测失败: {e}")
            return []

    # ==================== 社区检测 (Louvain/Label Propagation) ====================

    async def detect_communities(self) -> CommunityResult:
        if not self.driver:
            return CommunityResult(community_count=0, communities={}, community_labels={}, community_sizes={})
        try:
            async with self.driver.session() as session:
                try:
                    await session.run("CALL gds.graph.drop('kb_graph', false)")
                except Exception:
                    pass

                try:
                    await session.run(
                        """
                        CALL gds.graph.project(
                            'kb_graph',
                            ['Entity', 'Concept', 'Event', 'Document', 'Function', 'Class', 'Module', 'Algorithm', 'File'],
                            {
                                RELATES: {orientation: 'UNDIRECTED'},
                                SAME_AS: {orientation: 'UNDIRECTED'},
                                PART_OF: {orientation: 'UNDIRECTED'},
                                DEFINES_FUNCTION: {orientation: 'UNDIRECTED'},
                                DEFINES_CLASS: {orientation: 'UNDIRECTED'},
                                CALLS: {orientation: 'UNDIRECTED'},
                                IMPLEMENTS_ALGORITHM: {orientation: 'UNDIRECTED'},
                                IMPORTS: {orientation: 'UNDIRECTED'},
                                RELATED_TO_CONCEPT: {orientation: 'UNDIRECTED'},
                            }
                        )
                        """
                    )

                    result = await session.run(
                        """
                        CALL gds.louvain.write('kb_graph', {writeProperty: 'community_id'})
                        YIELD communityCount, modularity
                        RETURN communityCount, modularity
                        """
                    )
                    record = await result.single()
                    community_count = record["communityCount"] if record else 0
                    modularity = record["modularity"] if record else None
                except Exception:
                    return await self._label_propagation_fallback()

                community_result = await session.run(
                    """
                    MATCH (n)
                    WHERE n.community_id IS NOT NULL
                    RETURN n.community_id as cid, collect(n.name) as members,
                           count(n) as size
                    ORDER BY size DESC
                    LIMIT 20
                    """
                )
                communities = {}
                community_sizes = {}
                async for row in community_result:
                    cid = row["cid"]
                    communities[cid] = row["members"]
                    community_sizes[cid] = row["size"]

                community_labels = {}
                for cid, members in communities.items():
                    label = members[0][:30] if members else str(cid)
                    community_labels[cid] = f"社区_{label}"

                try:
                    await session.run("CALL gds.graph.drop('kb_graph', false)")
                except Exception:
                    pass

                return CommunityResult(
                    community_count=community_count,
                    modularity=modularity,
                    communities=communities,
                    community_labels=community_labels,
                    community_sizes=community_sizes,
                )

        except Exception as e:
            logger.warning(f"社区检测失败: {e}")
            return await self._label_propagation_fallback()

    async def _label_propagation_fallback(self) -> CommunityResult:
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (n)-[r]-(m)
                    WITH n, collect(distinct id(m)) as neighbors
                    SET n._tmp_community = toInteger(rand() * 1000)
                    """
                )
                for _ in range(5):
                    await session.run(
                        """
                        MATCH (n)-[r]-(m)
                        WITH n, collect(distinct m._tmp_community) as neighbor_comms
                        WITH n, [val IN neighbor_comms WHERE val IS NOT NULL] as valid_comms
                        WITH n, valid_comms,
                             CASE WHEN size(valid_comms) > 0
                                  THEN reduce(acc = {}, c IN valid_comms | acc + {c: coalesce(acc[c], 0) + 1})
                                  ELSE {}
                             END as freq
                        WITH n, reduce(best = [null, 0], k IN keys(freq) | CASE WHEN freq[k] > best[1] THEN [toInteger(k), freq[k]] ELSE best END) as best
                        SET n._tmp_community = best[0]
                        """
                    )

                result = await session.run(
                    """
                    MATCH (n)
                    WHERE n._tmp_community IS NOT NULL
                    RETURN n._tmp_community as cid, collect(n.name) as members, count(n) as size
                    ORDER BY size DESC
                    LIMIT 20
                    """
                )
                communities = {}
                community_labels = {}
                community_sizes = {}
                async for row in result:
                    cid = row["cid"]
                    communities[cid] = row["members"]
                    community_sizes[cid] = row["size"]
                    label = row["members"][0][:30] if row["members"] else str(cid)
                    community_labels[cid] = f"社区_{label}"

                return CommunityResult(
                    community_count=len(communities),
                    communities=communities,
                    community_labels=community_labels,
                    community_sizes=community_sizes,
                )
        except Exception as e:
            logger.warning(f"Label Propagation 回退失败: {e}")
            return CommunityResult(community_count=0, communities={}, community_labels={}, community_sizes={})

    # ==================== 推理规则 ====================

    async def apply_inference_rules(self) -> list[RuleInference]:
        if not self.driver:
            return []
        results = []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (x)-[r1:RELATES]->(y)-[r2:RELATES]->(z)
                    WHERE r1.relation_type = 'IS_A' AND r2.relation_type = 'CAUSES'
                      AND NOT (x)-[:RELATES {relation_type: 'CAUSES'}]->(z)
                    RETURN x.name as subject, z.name as object,
                           r1.relation as rel1, r2.relation as rel2
                    LIMIT 30
                    """
                )
                new_triples = []
                async for record in result:
                    new_triples.append(KnowledgeTriple(
                        subject=record["subject"],
                        relation=f"推理: {record['rel1']} → {record['rel2']}",
                        object=record["object"],
                        relation_type=RelationType.CAUSES,
                        confidence=0.5,
                    ))
                if new_triples:
                    await self.create_triples(new_triples)
                    results.append(RuleInference(
                        rule_name="IS_A_CAUSES传导",
                        rule_description="若 X IS_A Y，且 Y CAUSES Z，则 X 也可能 CAUSES Z",
                        inferred_count=len(new_triples),
                        new_triples=new_triples,
                    ))

                result2 = await session.run(
                    """
                    MATCH (a)-[r1:RELATES]->(b)
                    WHERE r1.relation_type = 'PART_OF'
                    WITH a, collect(b) as parts
                    MATCH (part)-[r2:RELATES]->(c)
                    WHERE part IN parts AND r2.relation_type IN ['CAUSES', 'INDICATES', 'DEPENDS_ON']
                      AND NOT (a)-[:RELATES {relation_type: r2.relation_type}]->(c)
                    RETURN a.name as subject, c.name as object, r2.relation as rel,
                           r2.relation_type as rel_type
                    LIMIT 30
                    """
                )
                new_triples2 = []
                async for record in result2:
                    rel_type_str = record.get("rel_type", "RELATED_TO")
                    new_triples2.append(KnowledgeTriple(
                        subject=record["subject"],
                        relation=f"推理: 组成部分{record['rel']}",
                        object=record["object"],
                        relation_type=RelationType(rel_type_str) if rel_type_str in KNOWN_RELATION_TYPES else RelationType.RELATED_TO,
                        confidence=0.4,
                    ))
                if new_triples2:
                    await self.create_triples(new_triples2)
                    results.append(RuleInference(
                        rule_name="PART_OF传导",
                        rule_description="若 A PARTOF B，且 A 的某部分 CAUSES/INDICATES/DEPENDS_ON C，则 B 也可能有此关系",
                        inferred_count=len(new_triples2),
                        new_triples=new_triples2,
                    ))

        except Exception as e:
            logger.warning(f"推理规则应用失败: {e}")

        return results

    # ==================== Cypher 查询执行 ====================

    async def execute_cypher(self, query: str, params: dict = None) -> list[dict]:
        if not self.driver:
            return []
        query_upper = query.strip().upper()

        safe_start_keywords = ["MATCH", "RETURN", "WITH", "OPTIONAL"]
        if not any(query_upper.startswith(kw) for kw in safe_start_keywords):
            raise ValueError("仅允许以 MATCH / RETURN / WITH / OPTIONAL 开头的只读查询")

        forbidden_pattern = re.compile(
            r'\b(DELETE|DETACH\s+DELETE|REMOVE|SET\s|CREATE|MERGE|DROP|CALL|LOAD\s+CSV|FOREACH|UNWIND)\b',
            re.IGNORECASE
        )
        match = forbidden_pattern.search(query)
        if match:
            raise ValueError(f"禁止操作: {match.group(1)}")

        try:
            async with self.driver.session() as session:
                result = await session.run(query, params or {})
                return await result.data()
        except Exception as e:
            logger.warning(f"Cypher 查询失败: {e}")
            raise

    # ==================== 置信度传播（增强版） ====================

    async def propagate_confidence(self, iterations: int = 3, damping: float = 0.85) -> dict[str, float]:
        if not self.driver:
            return {}
        try:
            async with self.driver.session() as session:
                await session.run(
                    "MATCH (n) WHERE n.type IS NOT NULL SET n.prop_conf = coalesce(n.prop_conf, 0.5)"
                )
                for _ in range(iterations):
                    await session.run(
                        f"""
                        MATCH (e)
                        WHERE e.type IS NOT NULL
                        OPTIONAL MATCH (e)<-[r:RELATES]-(n)
                        WHERE n.type IS NOT NULL
                        WITH e, avg(coalesce(n.prop_conf, 0.5)) as avg_neighbor
                        SET e.prop_conf = {1 - damping} * 0.5 + {damping} * coalesce(avg_neighbor, 0.5)
                        """
                    )
                result = await session.run(
                    "MATCH (e) WHERE e.type IS NOT NULL RETURN e.name as name, e.prop_conf as conf"
                )
                prop_map = {}
                async for record in result:
                    prop_map[record["name"]] = record["conf"]
                return prop_map
        except Exception as e:
            logger.warning(f"置信度传播失败: {e}")
            return {}

    # ==================== 版本管理（逻辑删除） ====================

    async def soft_delete_knowledge(self, knowledge_id: str) -> dict:
        if not self.driver:
            return {"status": "unavailable"}
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MATCH (k:KnowledgeAtom {knowledge_id: $kid})
                    SET k.is_active = false, k.deactivated_at = timestamp()
                    """,
                    kid=knowledge_id,
                )
                return {"status": "deactivated", "knowledge_id": knowledge_id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def create_new_version(self, old_knowledge_id: str, new_knowledge_id: str, new_fact: str, version: int) -> dict:
        if not self.driver:
            return {"status": "unavailable"}
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MERGE (k:KnowledgeAtom {knowledge_id: $kid})
                    SET k.fact = $fact, k.version = $version,
                        k.is_active = true, k.created_at = timestamp()
                    WITH k
                    MATCH (old:KnowledgeAtom {knowledge_id: $old_kid})
                    MERGE (old)-[r:REVISED_TO]->(k)
                    SET r.created_at = timestamp(), r.version_from = old.version,
                        r.version_to = $version
                    """,
                    kid=new_knowledge_id,
                    fact=new_fact,
                    version=version,
                    old_kid=old_knowledge_id,
                )
                return {"status": "versioned", "new_id": new_knowledge_id, "version": version}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ==================== 节点统计 ====================

    async def count_nodes(self) -> int:
        if not self.driver:
            return 0
        try:
            async with self.driver.session() as session:
                result = await session.run("MATCH (n) WHERE n.type IS NOT NULL RETURN count(n) as cnt")
                record = await result.single()
                return record["cnt"] if record else 0
        except Exception:
            return 0

    async def get_graph_stats(self) -> dict:
        if not self.driver:
            return {"node_count": 0, "edge_count": 0, "node_types": {}}
        try:
            async with self.driver.session() as session:
                node_result = await session.run(
                    "MATCH (n) WHERE n.type IS NOT NULL RETURN n.type as type, count(n) as cnt"
                )
                node_types = {}
                async for record in node_result:
                    node_types[record["type"]] = record["cnt"]

                edge_result = await session.run(
                    "MATCH ()-[r:RELATES]->() RETURN count(r) as cnt"
                )
                edge_record = await edge_result.single()
                edge_count = edge_record["cnt"] if edge_record else 0

                total_nodes = sum(node_types.values())

                return {
                    "node_count": total_nodes,
                    "edge_count": edge_count,
                    "node_types": node_types,
                }
        except Exception as e:
            logger.warning(f"图谱统计失败: {e}")
            return {"node_count": 0, "edge_count": 0, "node_types": {}}

    # ==================== 知识融合扫描 ====================

    async def find_similar_atoms(self, threshold: float = 0.95) -> list[dict]:
        if not self.driver:
            return []
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (k1:KnowledgeAtom {is_active: true})-[r1:EVIDENCED_BY]->(:Document)
                    MATCH (k2:KnowledgeAtom {is_active: true})
                    WHERE k1.knowledge_id < k2.knowledge_id AND k1.fact = k2.fact
                    RETURN k1.knowledge_id as kid1, k2.knowledge_id as kid2,
                           k1.fact as fact, k1.version as v1, k2.version as v2
                    LIMIT 20
                    """
                )
                pairs = []
                async for record in result:
                    pairs.append({
                        "knowledge_id_1": record["kid1"],
                        "knowledge_id_2": record["kid2"],
                        "fact": record.get("fact", ""),
                        "version_1": record.get("v1", 1),
                        "version_2": record.get("v2", 1),
                    })
                return pairs
        except Exception as e:
            logger.warning(f"相似知识扫描失败: {e}")
            return []

    async def delete_knowledge_node(self, kp_id: str):
        if not self.driver:
            return
        try:
            async with self.driver.session() as session:
                await session.run(
                    "MATCH (k:KnowledgeAtom {knowledge_id: $kid}) DETACH DELETE k",
                    kid=kp_id,
                )
                await session.run(
                    """
                    MATCH ()-[r:RELATES]->()
                    WHERE $kid IN split(coalesce(r.source_knowledge_id, ''), ',')
                    DELETE r
                    """,
                    kid=kp_id,
                )
                await session.run(
                    """
                    MATCH ()-[r:EVIDENCED_BY]->()
                    WHERE $kid IN split(coalesce(r.source_knowledge_id, ''), ',')
                    DELETE r
                    """,
                    kid=kp_id,
                )
                await self._cleanup_orphaned_nodes(session)
        except Exception:
            pass

    async def _cleanup_orphaned_nodes(self, session=None):
        if not self.driver:
            return
        try:
            if session is None:
                async with self.driver.session() as session:
                    await session.run(
                        """
                        MATCH (n)
                        WHERE n.type IS NOT NULL
                          AND NOT n:KnowledgeAtom
                          AND NOT (n)--()
                        DETACH DELETE n
                        """
                    )
            else:
                await session.run(
                    """
                    MATCH (n)
                    WHERE n.type IS NOT NULL
                      AND NOT n:KnowledgeAtom
                      AND NOT (n)--()
                    DETACH DELETE n
                    """
                )
        except Exception:
            pass

    async def delete_all(self) -> int:
        if not self.driver:
            return 0
        try:
            async with self.driver.session() as session:
                result = await session.run("MATCH (n) DETACH DELETE n RETURN count(n) as cnt")
                record = await result.single()
                cnt = record["cnt"] if record else 0
                logger.info(f"已清空 Neo4j 图谱: {cnt} 个节点")
                return cnt
        except Exception as e:
            logger.warning(f"Neo4j 清空失败: {e}")
            return 0

    async def create_triples_with_dedup(
        self, triples: list[KnowledgeTriple], source_chunk_id: Optional[str] = None,
        knowledge_points: list = None,
    ) -> dict:
        stats = {"created": 0, "merged": 0, "dedup_skipped": 0, "total": len(triples)}
        if not self.driver:
            logger.info("Neo4j 未连接，跳过三元组写入")
            return stats
        try:
            async with self.driver.session() as session:
                for triple in triples:
                    rel_type = triple.relation_type.value if triple.relation_type else _classify_relation_type(triple.relation)
                    chunk_id = triple.source_chunk_id or source_chunk_id
                    subj_type = _infer_node_type(triple.subject)
                    obj_type = _infer_node_type(triple.object)

                    exists_result = await session.run(
                        """
                        MATCH (s:%s {name: $subject})-[r:RELATES {relation: $relation, relation_type: $rel_type}]->(o:%s {name: $object})
                        RETURN count(r) as cnt
                        """ % (subj_type, obj_type),
                        subject=triple.subject, object=triple.object,
                        relation=triple.relation, rel_type=rel_type,
                    )
                    exists_record = await exists_result.single()

                    if exists_record and exists_record["cnt"] > 0:
                        await session.run(
                            f"""
                            MATCH (s:{subj_type} {{name: $subject}})-[r:RELATES {{relation: $relation, relation_type: $rel_type}}]->(o:{obj_type} {{name: $object}})
                            SET r.confidence = CASE
                                    WHEN $confidence > coalesce(r.confidence, 0) THEN $confidence
                                    ELSE r.confidence
                                END,
                                r.source_knowledge_id = CASE
                                    WHEN r.source_knowledge_id IS NULL OR r.source_knowledge_id = '' THEN $source_kid
                                    WHEN $source_kid = '' OR $source_kid IS NULL THEN r.source_knowledge_id
                                    WHEN $source_kid IN split(r.source_knowledge_id, ',') THEN r.source_knowledge_id
                                    ELSE r.source_knowledge_id + ',' + $source_kid
                                END,
                                r.updated_at = timestamp(),
                                r.merge_count = coalesce(r.merge_count, 1) + 1
                            SET s.updated_at = timestamp(),
                                o.updated_at = timestamp()
                            """,
                            subject=triple.subject, object=triple.object,
                            relation=triple.relation, rel_type=rel_type,
                            confidence=triple.confidence,
                            source_kid=triple.source_knowledge_id or "",
                        )
                        stats["merged"] += 1
                    else:
                        await session.run(
                            f"""
                            MERGE (s:{subj_type} {{name: $subject}})
                            SET s.type = $subj_type, s.updated_at = timestamp()
                            MERGE (o:{obj_type} {{name: $object}})
                            SET o.type = $obj_type, o.updated_at = timestamp()
                            CREATE (s)-[r:RELATES {{relation: $relation, relation_type: $rel_type}}]->(o)
                            SET r.confidence = $confidence,
                                r.source_knowledge_id = $source_kid,
                                r.source_chunk_id = $chunk_id,
                                r.created_at = timestamp(),
                                r.updated_at = timestamp(),
                                r.merge_count = 1
                            """,
                            subject=triple.subject, object=triple.object,
                            relation=triple.relation, rel_type=rel_type,
                            confidence=triple.confidence,
                            source_kid=triple.source_knowledge_id or "",
                            chunk_id=chunk_id or "",
                        )
                        stats["created"] += 1

            logger.info(f"[去重] 三元组写入完成: 创建{stats['created']}, 合并{stats['merged']}, 总计{stats['total']}")
            return stats
        except Exception as e:
            logger.error(f"Neo4j 去重写入失败: {e}")
            return stats

    async def merge_similar_triples(
        self, fact_text: str, source_knowledge_id: str, confidence: float,
        similarity_threshold: float = 0.85
    ) -> Optional[dict]:
        if not self.driver:
            return None
        try:
            async with self.driver.session() as session:
                existing_result = await session.run(
                    """
                    MATCH (s)-[r:RELATES]->(o)
                    WHERE r.source_knowledge_id IS NOT NULL
                      AND r.source_knowledge_id <> $source_kid
                    RETURN s.name as subject, r.relation as relation, o.name as object,
                           r.relation_type as rel_type, r.confidence as confidence,
                           r.source_knowledge_id as source_kid, r.merge_count as merges
                    LIMIT 50
                    """,
                    source_kid=source_knowledge_id,
                )
                candidates = []
                async for record in existing_result:
                    candidates.append({
                        "subject": record["subject"],
                        "relation": record["relation"],
                        "object": record["object"],
                        "rel_type": record.get("rel_type", "RELATED_TO"),
                        "confidence": record.get("confidence", 0.5),
                        "source_kid": record.get("source_kid", ""),
                        "merges": record.get("merges", 1),
                    })

                simplified_fact = re.sub(r'[，,。；;：:\s]+', '', fact_text.lower())[:200]
                best_match = None
                best_score = 0.0
                for c in candidates:
                    candidate_fact = f"{c['subject']}{c['relation']}{c['object']}"
                    candidate_simplified = re.sub(r'[，,。；;：:\s]+', '', candidate_fact.lower())[:200]
                    overlap = len(set(simplified_fact) & set(candidate_simplified))
                    max_len = max(len(simplified_fact), len(candidate_simplified))
                    score = overlap / max_len if max_len > 0 else 0.0
                    if score > best_score and score > 0.6:
                        best_score = score
                        best_match = c

                if best_match and best_score >= similarity_threshold:
                    await session.run(
                        """
                        MATCH (s {name: $subj_name})-[r:RELATES {relation: $rel, relation_type: $rel_type}]->(o {name: $obj_name})
                        SET r.merge_count = coalesce(r.merge_count, 1) + 1,
                            r.source_knowledge_id = CASE
                                WHEN $source_kid IN split(coalesce(r.source_knowledge_id, ''), ',') THEN r.source_knowledge_id
                                ELSE coalesce(r.source_knowledge_id, '') + ',' + $source_kid
                            END,
                            r.updated_at = timestamp(),
                            r.confidence = CASE
                                WHEN $confidence > coalesce(r.confidence, 0) THEN $confidence
                                ELSE r.confidence
                            END
                        """,
                        subj_name=best_match["subject"],
                        rel=best_match["relation"],
                        rel_type=best_match.get("rel_type", "RELATED_TO"),
                        obj_name=best_match["object"],
                        source_kid=source_knowledge_id,
                        confidence=confidence,
                    )
                    logger.info(f"[去重] 语义合并三元组: {fact_text[:60]} -> {best_match['subject']}-{best_match['relation']}->{best_match['object']} (相似度:{best_score:.2f})")
                    return {"action": "merged", "similarity": best_score, "merged_with": best_match}

            return None
        except Exception as e:
            logger.warning(f"语义三元组合并失败: {e}")
            return None