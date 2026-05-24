import ast
import os
import re
import logging
from typing import Optional

import tree_sitter

logger = logging.getLogger(__name__)

_LANGUAGE_CACHE: dict[str, "tree_sitter.Language"] = {}


def _get_tree_sitter_language(lang_name: str):
    if lang_name in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[lang_name]
    try:
        if lang_name == "python":
            import tree_sitter_python as m; lang = tree_sitter.Language(m.language())
        elif lang_name in ("javascript", "typescript"):
            import tree_sitter_javascript as m; lang = tree_sitter.Language(m.language())
        elif lang_name == "go":
            import tree_sitter_go as m; lang = tree_sitter.Language(m.language())
        elif lang_name == "java":
            import tree_sitter_java as m; lang = tree_sitter.Language(m.language())
        elif lang_name in ("cpp", "c"):
            import tree_sitter_cpp as m; lang = tree_sitter.Language(m.language())
        elif lang_name == "rust":
            import tree_sitter_rust as m; lang = tree_sitter.Language(m.language())
        elif lang_name == "csharp":
            import tree_sitter_c_sharp as m; lang = tree_sitter.Language(m.language())
        else:
            return None
        _LANGUAGE_CACHE[lang_name] = lang
        return lang
    except Exception as e:
        logger.debug(f"tree-sitter 加载失败 ({lang_name}): {e}")
        return None


TREE_SITTER_NODE_CONFIG = {
    "python": {
        "functions": ["function_definition", "async_function_definition"],
        "classes": ["class_definition"],
        "imports": ["import_statement", "import_from_statement", "future_import_statement"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "if_statement", "try_statement", "with_statement",
                       "raise_statement", "assert_statement", "yield", "await"],
    },
    "javascript": {
        "functions": ["function_declaration", "generator_function_declaration",
                      "arrow_function", "method_definition"],
        "classes": ["class_declaration"],
        "imports": ["import_statement"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "if_statement", "try_statement", "switch_statement",
                       "throw_statement", "do_statement"],
    },
    "typescript": {
        "functions": ["function_declaration", "generator_function_declaration",
                      "arrow_function", "method_definition", "method_signature"],
        "classes": ["class_declaration"],
        "imports": ["import_statement"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "if_statement", "try_statement", "switch_statement",
                       "throw_statement", "do_statement"],
    },
    "go": {
        "functions": ["function_declaration", "method_declaration"],
        "classes": ["type_declaration"],
        "imports": ["import_declaration"],
        "stmt_types": ["return_statement", "for_statement", "if_statement",
                       "switch_statement", "select_statement", "go_statement", "defer_statement"],
    },
    "java": {
        "functions": ["method_declaration", "constructor_declaration"],
        "classes": ["class_declaration", "interface_declaration", "enum_declaration"],
        "imports": ["import_declaration", "package_declaration"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "if_statement", "try_statement", "switch_expression",
                       "throw_statement", "synchronized_statement"],
    },
    "cpp": {
        "functions": ["function_definition", "template_declaration"],
        "classes": ["class_specifier", "struct_specifier"],
        "imports": ["preproc_include"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "if_statement", "try_statement", "switch_statement",
                       "throw_statement"],
    },
    "c": {
        "functions": ["function_definition"],
        "classes": [],
        "imports": ["preproc_include"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "if_statement", "switch_statement"],
    },
    "rust": {
        "functions": ["function_item"],
        "classes": ["struct_item", "enum_item", "trait_item", "impl_item"],
        "imports": ["use_declaration"],
        "stmt_types": ["return_expression", "for_expression", "while_expression",
                       "if_expression", "loop_expression", "match_expression"],
    },
    "csharp": {
        "functions": ["method_declaration", "constructor_declaration", "local_function_statement"],
        "classes": ["class_declaration", "interface_declaration", "struct_declaration", "enum_declaration"],
        "imports": ["using_directive"],
        "stmt_types": ["return_statement", "for_statement", "while_statement",
                       "foreach_statement", "if_statement", "try_statement",
                       "switch_statement", "throw_statement"],
    },
}


def _ts_extract_name(node) -> Optional[str]:
    name_child = node.child_by_field_name("name")
    if name_child:
        return name_child.text.decode("utf-8", errors="replace")
    for child in node.children:
        if child.type == "identifier" or child.type == "type_identifier":
            text = child.text.decode("utf-8", errors="replace")
            if text:
                return text
    if node.type == "function_definition":
        for child in node.children:
            if child.type == "function_declarator":
                for gc in child.children:
                    if gc.type in ("identifier", "field_identifier", "qualified_identifier"):
                        return gc.text.decode("utf-8", errors="replace")
    if node.type in ("type_declaration", "type_spec"):
        for child in node.children:
            if child.type == "type_spec":
                result = _ts_extract_name(child)
                if result:
                    return result
            if child.type == "type_identifier":
                return child.text.decode("utf-8", errors="replace")
    if node.type == "template_declaration":
        for child in node.children:
            if child.type == "function_definition":
                result = _ts_extract_name(child)
                if result:
                    return result
    if node.type == "class_specifier":
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8", errors="replace")
    return None


def _ts_extract_params(node) -> list[str]:
    params_child = node.child_by_field_name("parameters")
    if params_child is None or params_child is None:
        return []
    params = []
    for child in params_child.children:
        if child.type == "identifier":
            params.append(child.text.decode("utf-8", errors="replace"))
        elif child.type in ("required_parameter", "optional_parameter"):
            ident = None
            for c in child.children:
                if c.type == "identifier":
                    ident = c.text.decode("utf-8", errors="replace")
                    break
            if ident:
                params.append(ident)
    return params


def _ts_extract_calls(node) -> list[str]:
    calls = set()
    def _walk(n):
        if n.type == "call_expression":
            func_node = n.child_by_field_name("function")
            if func_node:
                name = None
                if func_node.type == "identifier":
                    name = func_node.text.decode("utf-8", errors="replace")
                elif func_node.type == "member_expression":
                    obj = func_node.child_by_field_name("object")
                    prop = func_node.child_by_field_name("property")
                    if obj and prop:
                        obj_name = obj.text.decode("utf-8", errors="replace")
                        prop_name = prop.text.decode("utf-8", errors="replace")
                        name = f"{obj_name}.{prop_name}"
                elif func_node.type == "field_expression":
                    parts = []
                    f = func_node
                    while f and f.type == "field_expression":
                        val = f.child_by_field_name("field")
                        if val:
                            parts.insert(0, val.text.decode("utf-8", errors="replace"))
                        f = f.child_by_field_name("value")
                    if f and f.type == "identifier":
                        parts.insert(0, f.text.decode("utf-8", errors="replace"))
                    name = ".".join(parts)
                if name:
                    calls.add(name)
        for child in n.children:
            _walk(child)
    _walk(node)
    return list(calls)


def _ts_has_stmt_type(node, stmt_types: list[str]) -> dict:
    result = {}
    type_to_key = {
        "return_statement": "has_return", "for_statement": "has_loop",
        "while_statement": "has_loop", "if_statement": "has_condition",
        "try_statement": "has_try", "switch_statement": "has_switch",
        "throw_statement": "has_throw", "return_expression": "has_return",
        "for_expression": "has_loop", "while_expression": "has_loop",
        "if_expression": "has_condition", "loop_expression": "has_loop",
        "match_expression": "has_match", "foreach_statement": "has_loop",
        "go_statement": "has_concurrent", "select_statement": "has_select",
        "defer_statement": "has_defer", "with_statement": "has_with",
        "raise_statement": "has_raise", "assert_statement": "has_assert",
        "yield": "has_yield", "await": "has_async",
        "do_statement": "has_loop", "synchronized_statement": "has_sync",
    }
    def _walk(n):
        if n.type in stmt_types:
            key = type_to_key.get(n.type, f"has_{n.type}")
            result[key] = True
        for child in n.children:
            _walk(child)
    _walk(node)
    return result


def _analyze_with_tree_sitter(source_code: str, language: str) -> dict:
    ts_lang = _get_tree_sitter_language(language)
    if ts_lang is None:
        return None

    parser = tree_sitter.Parser(ts_lang)
    tree = parser.parse(bytes(source_code, "utf-8"))
    root = tree.root_node

    config = TREE_SITTER_NODE_CONFIG.get(language, {})
    func_types = set(config.get("functions", []))
    class_types = set(config.get("classes", []))
    import_types = set(config.get("imports", []))
    stmt_types = config.get("stmt_types", [])

    functions = []
    classes = []
    imports = []
    top_level_vars = []
    loc = source_code.count("\n") + 1
    cursor = root.walk()

    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in func_types and node.id not in visited:
            visited.add(node.id)
            name = _ts_extract_name(node) or f"<anonymous_{len(functions)}>"
            func_info = {
                "name": name,
                "lineno": node.start_point[0] + 1,
                "end_lineno": node.end_point[0] + 1,
                "args": _ts_extract_params(node),
                "arg_count": 0,
                "called_functions": [],
            }
            func_info["arg_count"] = len(func_info["args"])
            func_info["called_functions"] = _ts_extract_calls(node)
            func_info["line_count"] = max(1, node.end_point[0] - node.start_point[0] + 1)
            stmt_info = _ts_has_stmt_type(node, stmt_types)
            func_info.update(stmt_info)
            functions.append(func_info)

        elif node.type in class_types and node.id not in visited:
            visited.add(node.id)
            name = _ts_extract_name(node) or f"<anonymous_class_{len(classes)}>"
            base_names = []
            bases = node.child_by_field_name("bases") or node.child_by_field_name("superclasses")
            if bases:
                for b in bases.children:
                    bn = _ts_extract_name(b)
                    if bn:
                        base_names.append(bn)
            class_methods = []
            class_start = node.start_point[0] + 1
            class_end = node.end_point[0] + 1
            for child in node.children:
                if child.type in func_types:
                    mn = _ts_extract_name(child)
                    if mn:
                        class_methods.append(mn)
            classes.append({
                "name": name,
                "lineno": class_start,
                "end_lineno": class_end,
                "bases": base_names,
                "methods": class_methods,
                "line_count": max(1, class_end - class_start + 1),
            })

        elif node.type in import_types:
            _ts_collect_import(node, imports, language)
        elif node.type in ("variable_declaration", "lexical_declaration", "let_declaration"):
            for child in node.children:
                if child.type in ("variable_declarator", "identifier"):
                    var_name = child.text.decode("utf-8", errors="replace")
                    var_name = var_name.split("=")[0].strip()
                    if var_name and not var_name.startswith("//"):
                        top_level_vars.append(var_name)

        for child in reversed(node.children):
            stack.append(child)

    return {
        "language": language,
        "loc": loc,
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "function_count": len(functions),
        "class_count": len(classes),
        "import_count": len(imports),
        "top_level_variables": list(set(top_level_vars))[:30],
    }


def _ts_collect_import(node, imports: list, language: str):
    text = node.text.decode("utf-8", errors="replace").strip()
    if language in ("javascript", "typescript"):
        source_node = node.child_by_field_name("source")
        if source_node:
            module = source_node.text.decode("utf-8", errors="replace").strip("\"'")
            spec = node.child_by_field_name("import_clause") or node.child_by_field_name("name")
            names = []
            if spec:
                names = [spec.text.decode("utf-8", errors="replace")]
            imports.append({"module": module, "names": names, "type": "import"})
    elif language == "python":
        module_name = None
        imported_names = []
        is_from = False
        for child in node.children:
            ctext = child.text.decode("utf-8", errors="replace")
            if child.type == "dotted_name":
                module_name = ctext
            elif child.type == "aliased_import":
                for gc in child.children:
                    if gc.type == "dotted_name":
                        imported_names.append(gc.text.decode("utf-8", errors="replace"))
                        break
            elif child.type == "identifier":
                pass
        if node.type == "import_from_statement":
            imports.append({
                "module": module_name or "",
                "names": [{"name": n} for n in imported_names],
                "type": "from_import",
            })
        else:
            if module_name:
                imports.append({"module": module_name, "type": "import"})
    elif language == "go":
        for child in node.children:
            if child.type == "import_spec" or child.type == "import_spec_list":
                for gc in child.children:
                    if gc.type == "import_spec":
                        path_node = gc.child_by_field_name("path") or gc
                        p = path_node.text.decode("utf-8", errors="replace").strip("\"'")
                        if p:
                            imports.append({"module": p, "type": "import"})
    elif language == "java":
        module_parts = []
        for child in node.children:
            if child.type in ("identifier", "scoped_identifier"):
                module_parts.append(child.text.decode("utf-8", errors="replace"))
        if module_parts:
            module = ".".join(module_parts)
            imports.append({"module": module, "type": "import"})
    elif language in ("cpp", "c"):
        path_node = node.child_by_field_name("path")
        if path_node:
            path = path_node.text.decode("utf-8", errors="replace").strip("\"'<>")
            imports.append({"module": path, "type": "include"})
    elif language == "rust":
        module_parts = []
        for child in node.children:
            if child.type in ("identifier", "scoped_identifier", "scoped_type_identifier", "use_as_clause"):
                module_parts.append(child.text.decode("utf-8", errors="replace"))
        if module_parts:
            imports.append({"module": "::".join(module_parts), "type": "use"})
    elif language == "csharp":
        name_node = node.child_by_field_name("name") or node
        import_name = name_node.text.decode("utf-8", errors="replace").replace("using", "", 1).strip().rstrip(";")
        if import_name:
            imports.append({"module": import_name, "type": "using"})

CODE_SEMANTIC_PROMPT = """你是一位资深软件工程师。请分析以下代码，输出一个JSON对象，包含：

1. summary: 代码的整体功能（一句话，简洁明确）
2. algorithm: 使用的核心算法或设计模式（如"二分查找"、"工厂模式"、"事件驱动"），若无则填 null
3. business_logic: 代码所实现的业务规则或逻辑流程（分步骤描述，每步一句话）
4. inputs: 输入数据格式和含义
5. outputs: 输出数据格式和含义
6. edge_cases: 处理了哪些边界情况（列表）
7. related_concepts: 与此代码相关的技术概念或领域术语（关键词列表）
8. code_quality_notes: 代码质量评价（一句话）

只输出一个JSON对象，不要额外解释。

代码：
{code}"""


def detect_language(file_path: str) -> str:
    ext_map = {
        ".py": "python",
        ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust",
        ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".rb": "ruby", ".php": "php",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".lua": "lua", ".sh": "bash", ".sql": "sql",
        ".r": "r", ".m": "matlab", ".jl": "julia",
        ".html": "html", ".css": "css", ".scss": "scss",
        ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".xml": "xml", ".toml": "toml", ".ini": "ini",
        ".ipynb": "python",
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext_map.get(ext, "unknown")


def analyze_python_ast(source_code: str) -> dict:
    functions = []
    classes = []
    imports = []
    top_level_vars = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return {"error": f"语法解析失败: {e}", "functions": [], "classes": [], "imports": []}

    class FuncVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "alias": alias.asname,
                    "type": "import",
                })

        def visit_ImportFrom(self, node):
            module = node.module or ""
            names = [{"name": a.name, "alias": a.asname} for a in node.names]
            imports.append({
                "module": module,
                "names": names,
                "type": "from_import",
            })

        def visit_FunctionDef(self, node):
            func_info = {
                "name": node.name,
                "lineno": node.lineno,
                "args": [a.arg for a in node.args.args],
                "arg_count": len(node.args.args),
                "return_type": ast.unparse(node.returns) if node.returns else None,
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "docstring": ast.get_docstring(node),
                "called_functions": [],
            }
            func_info["called_functions"] = list(_find_calls(node))
            func_info["has_loop"] = _has_node_of_type(node, (ast.For, ast.While))
            func_info["has_condition"] = _has_node_of_type(node, ast.If)
            func_info["has_try"] = _has_node_of_type(node, ast.Try)
            func_info["line_count"] = max(1, node.end_lineno - node.lineno + 1 if node.end_lineno else 1)
            functions.append(func_info)

        def visit_AsyncFunctionDef(self, node):
            func_info = {
                "name": node.name,
                "lineno": node.lineno,
                "args": [a.arg for a in node.args.args],
                "arg_count": len(node.args.args),
                "return_type": ast.unparse(node.returns) if node.returns else None,
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "docstring": ast.get_docstring(node),
                "called_functions": list(_find_calls(node)),
                "async": True,
            }
            func_info["has_loop"] = _has_node_of_type(node, (ast.For, ast.While))
            func_info["has_condition"] = _has_node_of_type(node, ast.If)
            func_info["has_try"] = _has_node_of_type(node, ast.Try)
            func_info["line_count"] = max(1, node.end_lineno - node.lineno + 1 if node.end_lineno else 1)
            functions.append(func_info)

        def visit_ClassDef(self, node):
            class_info = {
                "name": node.name,
                "lineno": node.lineno,
                "bases": [ast.unparse(b) for b in node.bases],
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "docstring": ast.get_docstring(node),
                "methods": [],
                "line_count": max(1, node.end_lineno - node.lineno + 1 if node.end_lineno else 1),
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = item.name
                    class_info["methods"].append(method_name)
            classes.append(class_info)

        def visit_Assign(self, node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_vars.append(target.id)

    _find_calls_cache = {}
    def _find_calls(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    yield child.func.id
                elif isinstance(child.func, ast.Attribute):
                    parts = []
                    obj = child.func
                    while isinstance(obj, ast.Attribute):
                        parts.append(obj.attr)
                        obj = obj.value
                    if isinstance(obj, ast.Name):
                        parts.append(obj.id)
                        yield ".".join(reversed(parts))

    def _has_node_of_type(node, types):
        for child in ast.walk(node):
            if isinstance(child, types):
                return True
        return False

    visitor = FuncVisitor()
    visitor.visit(tree)

    loc = source_code.count("\n") + 1
    return {
        "language": "python",
        "loc": loc,
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "top_level_variables": list(set(top_level_vars))[:20],
        "function_count": len(functions),
        "class_count": len(classes),
        "import_count": len(imports),
    }


def analyze_generic(source_code: str, language: str) -> dict:
    functions = []
    imports = []
    lines = source_code.split("\n")

    if language in ("javascript", "typescript", "go", "rust", "java", "cpp", "csharp", "kotlin"):
        func_patterns = [
            (rf"^(?:(?:export\s+)?(?:async\s+)?function\s+(\w+))", language in ("javascript", "typescript")),
            (rf"^(?:public|private|protected|static)?\s*(?:async\s+)?(?:[\w<>\[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{{\s*$", language in ("java", "cpp", "csharp", "kotlin")),
            (rf"^(?:pub\s+)?fn\s+(\w+)", language in ("rust",)),
            (rf"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", language in ("go",)),
        ]
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            for pattern, enabled in func_patterns:
                if enabled:
                    m = re.match(pattern, line_stripped)
                    if m:
                        func_name = m.group(1)
                        if func_name not in ("if", "for", "while", "switch", "catch", "with"):
                            functions.append({
                                "name": func_name,
                                "lineno": line_idx + 1,
                            })
                        break

    import_patterns = {
        "javascript": [(r'(?:import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"])', "from_import"),
                       (r'(?:const\s+(?:\{([^}]+)\}|(\w+))\s*=\s*require\s*\([\'"]([^\'"]+)[\'"]\))', "require"),
                       (r'(?:import\s+[\'"]([^\'"]+)[\'"])', "import"),],
        "typescript": [(r'(?:import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"])', "from_import")],
        "go": [(r'import\s+[\'"]([^\'"]+)[\'"]', "import")],
        "rust": [(r'use\s+([\w:]+)', "use")],
    }

    for key, patterns in import_patterns.items():
        if language == key:
            for line in lines:
                for pat, itype in patterns:
                    m = re.match(r"^\s*" + pat, line.strip())
                    if m:
                        imports.append({"module": m.group(3) if m.lastindex >= 3 else m.group(1), "type": itype})

    return {
        "language": language,
        "loc": len(lines),
        "functions": functions,
        "imports": imports,
        "function_count": len(functions),
        "import_count": len(imports),
    }


def compute_complexity(source_code: str, language: str) -> list[dict]:
    results = []
    if language == "python":
        try:
            from radon.complexity import cc_visit
            from radon.raw import analyze
            raw = analyze(source_code)
            blocks = cc_visit(source_code)
            results = [
                {
                    "name": b.name,
                    "type": b.type,
                    "complexity": b.complexity,
                    "lineno": b.lineno,
                    "rank": chr(64 + min(6, b.rank.upper() if hasattr(b, 'rank') else 1)),
                }
                for b in blocks
            ]
        except Exception as e:
            logger.debug(f"复杂度计算失败: {e}")
    return results


def build_call_graph(functions: list[dict]) -> dict:
    graph = {"nodes": [], "edges": []}
    func_names = {f["name"] for f in functions}
    for f in functions:
        graph["nodes"].append(f["name"])
        for called in f.get("called_functions", []):
            if called in func_names:
                graph["edges"].append({"from": f["name"], "to": called})
    return graph


def extract_from_ipynb(file_path: str) -> str:
    import json
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        nb = json.load(f)
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            stripped = src.strip()
            if stripped:
                cells.append(stripped)
    return "\n# --- notebook cell ---\n".join(cells)


def analyze_code_file(file_path: str) -> dict:
    language = detect_language(file_path)
    if file_path.endswith(".ipynb"):
        source = extract_from_ipynb(file_path)
        language = "python"
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

    if language == "python":
        try:
            ast_result = analyze_python_ast(source)
        except Exception:
            lines = source.split("\n")
            ast_result = {
                "language": "python",
                "loc": len(lines),
                "functions": [],
                "classes": [],
                "imports": [],
                "function_count": 0,
                "class_count": 0,
                "import_count": 0,
                "error": "AST parse failed",
            }
    elif language in TREE_SITTER_NODE_CONFIG:
        try:
            ts_result = _analyze_with_tree_sitter(source, language)
            if ts_result is not None:
                ast_result = ts_result
            else:
                ast_result = analyze_generic(source, language)
        except Exception:
            logger.warning(f"tree-sitter 解析失败 ({language})，回退到正则解析")
            ast_result = analyze_generic(source, language)
    else:
        ast_result = analyze_generic(source, language)

    complexity = compute_complexity(source, language)

    call_graph = build_call_graph(ast_result.get("functions", []))

    return {
        **ast_result,
        "complexity": complexity,
        "call_graph": call_graph,
    }