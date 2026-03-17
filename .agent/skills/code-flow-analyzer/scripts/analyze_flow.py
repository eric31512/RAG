#!/usr/bin/env python3
"""
Code Flow Analyzer
==================
Statically analyze Python source files using the ast module,
trace function-level call chains, and generate a Mermaid flowchart
with a comprehensive Markdown report.

Usage:
    python analyze_flow.py <source_dir> [--output <path>] [--entry <file>] [--recursive]
"""

import ast
import os
import sys
import argparse
import textwrap
from collections import defaultdict
from pathlib import Path


# ─── AST Helpers ───────────────────────────────────────────────────────────────

class FunctionCallVisitor(ast.NodeVisitor):
    """Extract all function/method call names from an AST node."""

    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        name = self._resolve_call_name(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)

    @staticmethod
    def _resolve_call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None


class FileAnalyzer:
    """Analyze a single Python file."""

    def __init__(self, filepath: str, base_dir: str):
        self.filepath = filepath
        self.basename = os.path.basename(filepath)
        self.module_name = Path(filepath).stem
        self.base_dir = base_dir
        self.tree = None
        self.imports = []          # [(module, names)]
        self.local_imports = []    # [module_name] — imports of other local files
        self.classes = []          # [{name, methods: [{name, calls, lineno}], lineno}]
        self.functions = []        # [{name, calls, lineno, is_entry}]
        self.docstring = ""
        self.has_main_block = False

    def analyze(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            self.tree = ast.parse(source, filename=self.filepath)
        except SyntaxError as e:
            print(f"  ⚠ Skipping {self.basename}: {e}", file=sys.stderr)
            return False

        self.docstring = ast.get_docstring(self.tree) or ""
        self._extract_imports()
        self._extract_functions()
        self._extract_classes()
        self._check_main_block()
        return True

    # ── Imports ──

    def _extract_imports(self):
        local_py_stems = {
            Path(f).stem
            for f in os.listdir(self.base_dir)
            if f.endswith(".py") and f != self.basename
        }

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append((alias.name, alias.asname or alias.name))
                    if alias.name in local_py_stems:
                        self.local_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [a.name for a in node.names]
                self.imports.append((module, names))
                if module in local_py_stems:
                    self.local_imports.append(module)

    # ── Functions ──

    def _extract_functions(self):
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                visitor = FunctionCallVisitor()
                visitor.visit(node)
                self.functions.append({
                    "name": node.name,
                    "calls": visitor.calls,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", None),
                    "is_entry": False,
                    "docstring": ast.get_docstring(node) or "",
                    "args": self._format_args(node.args),
                })

    # ── Classes ──

    def _extract_classes(self):
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visitor = FunctionCallVisitor()
                        visitor.visit(item)
                        methods.append({
                            "name": item.name,
                            "calls": visitor.calls,
                            "lineno": item.lineno,
                            "docstring": ast.get_docstring(item) or "",
                            "args": self._format_args(item.args),
                        })
                self.classes.append({
                    "name": node.name,
                    "methods": methods,
                    "lineno": node.lineno,
                    "bases": [self._base_name(b) for b in node.bases],
                    "docstring": ast.get_docstring(node) or "",
                })

    # ── Main block ──

    def _check_main_block(self):
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    self.has_main_block = True
                    # Mark called functions as entry points
                    visitor = FunctionCallVisitor()
                    for child in node.body:
                        visitor.visit(child)
                    entry_calls = set(visitor.calls)
                    for fn in self.functions:
                        if fn["name"] in entry_calls:
                            fn["is_entry"] = True

    # ── Utilities ──

    @staticmethod
    def _format_args(args_node):
        params = []
        for a in args_node.args:
            if a.arg != "self" and a.arg != "cls":
                params.append(a.arg)
        return params

    @staticmethod
    def _base_name(node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{FileAnalyzer._base_name(node.value)}.{node.attr}"
        return "?"


# ─── Project Analyzer ─────────────────────────────────────────────────────────

class ProjectAnalyzer:
    """Analyze all Python files in a directory."""

    def __init__(self, source_dir: str, recursive: bool = False, entry_file: str = None):
        self.source_dir = os.path.abspath(source_dir)
        self.recursive = recursive
        self.entry_file = entry_file
        self.files: list[FileAnalyzer] = []
        # Registry: function_name -> module_name (for cross-module resolution)
        self.fn_registry: dict[str, str] = {}

    def analyze(self):
        py_files = self._discover_py_files()
        if not py_files:
            print(f"Error: No .py files found in {self.source_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Analyzing {len(py_files)} Python files in {self.source_dir}...")
        for fpath in sorted(py_files):
            fa = FileAnalyzer(fpath, self.source_dir)
            if fa.analyze():
                self.files.append(fa)
                # Register functions
                for fn in fa.functions:
                    self.fn_registry[fn["name"]] = fa.module_name
                for cls in fa.classes:
                    for method in cls["methods"]:
                        qualified = f"{cls['name']}.{method['name']}"
                        self.fn_registry[qualified] = fa.module_name
                print(f"  ✓ {fa.basename}: {len(fa.functions)} functions, {len(fa.classes)} classes")

    def _discover_py_files(self):
        files = []
        if self.recursive:
            for root, _, filenames in os.walk(self.source_dir):
                for f in filenames:
                    if f.endswith(".py") and not f.startswith("__"):
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(self.source_dir):
                if f.endswith(".py") and not f.startswith("__"):
                    files.append(os.path.join(self.source_dir, f))
        return files

    def _get_entry_files(self):
        if self.entry_file:
            return [f for f in self.files if f.basename == self.entry_file]
        return [f for f in self.files if f.has_main_block]

    # ─── Report Generation ─────────────────────────────────────────────────

    def generate_report(self) -> str:
        md = []
        md.append("# Code Flow Analysis Report\n")
        md.append(f"**Source Directory**: `{self.source_dir}`  ")
        md.append(f"**Files Analyzed**: {len(self.files)}  ")
        entry_files = self._get_entry_files()
        if entry_files:
            md.append(f"**Entry Point(s)**: {', '.join(f'`{f.basename}`' for f in entry_files)}  ")
        md.append("")
        md.append("---\n")

        # 1. Project Overview
        md.append("## Project Overview\n")
        md.append("| File | Functions | Classes | Entry Point | Description |")
        md.append("|------|:---------:|:-------:|:-----------:|-------------|")
        for fa in self.files:
            desc = fa.docstring.split("\n")[0][:80] if fa.docstring else "—"
            entry = "✅" if fa.has_main_block else ""
            md.append(f"| `{fa.basename}` | {len(fa.functions)} | {len(fa.classes)} | {entry} | {desc} |")
        md.append("")

        # 2. Module Dependency Graph
        md.append("## Module Dependency Graph\n")
        md.append(self._gen_module_dep_mermaid())
        md.append("")

        # 3. Function Call Flow
        md.append("## Function Call Flow\n")
        md.append(self._gen_function_flow_mermaid())
        md.append("")

        # 4. Per-file Details
        md.append("## File Details\n")
        for fa in self.files:
            md.append(f"### `{fa.basename}`\n")
            if fa.docstring:
                md.append(f"> {fa.docstring.split(chr(10))[0]}\n")

            # Imports
            if fa.local_imports:
                md.append(f"**Local Imports**: {', '.join(f'`{m}`' for m in fa.local_imports)}\n")

            # Functions
            if fa.functions:
                md.append("**Functions:**\n")
                md.append("| Function | Line | Calls | Description |")
                md.append("|----------|:----:|-------|-------------|")
                for fn in fa.functions:
                    calls_str = ", ".join(f"`{c}`" for c in fn["calls"][:8])
                    if len(fn["calls"]) > 8:
                        calls_str += f" +{len(fn['calls'])-8} more"
                    desc = fn["docstring"].split("\n")[0][:60] if fn["docstring"] else "—"
                    entry_mark = " 🚀" if fn["is_entry"] else ""
                    md.append(f"| `{fn['name']}`{entry_mark} | {fn['lineno']} | {calls_str} | {desc} |")
                md.append("")

            # Classes
            for cls in fa.classes:
                bases = f" ({', '.join(cls['bases'])})" if cls['bases'] else ""
                md.append(f"**Class `{cls['name']}`{bases}** (line {cls['lineno']})\n")
                if cls["docstring"]:
                    md.append(f"> {cls['docstring'].split(chr(10))[0]}\n")
                if cls["methods"]:
                    md.append("| Method | Line | Calls |")
                    md.append("|--------|:----:|-------|")
                    for m in cls["methods"]:
                        calls_str = ", ".join(f"`{c}`" for c in m["calls"][:6])
                        if len(m["calls"]) > 6:
                            calls_str += f" +{len(m['calls'])-6} more"
                        md.append(f"| `{m['name']}` | {m['lineno']} | {calls_str} |")
                    md.append("")

        return "\n".join(md)

    # ─── Mermaid Diagram Generators ────────────────────────────────────────

    def _gen_module_dep_mermaid(self) -> str:
        lines = ["```mermaid", "graph LR"]
        # Define nodes
        for fa in self.files:
            style = "([" if fa.has_main_block else "["
            end_style = "])" if fa.has_main_block else "]"
            lines.append(f'    {fa.module_name}{style}"{fa.basename}"{end_style}')
        # Define edges
        seen_edges = set()
        for fa in self.files:
            for imp in fa.local_imports:
                edge = (fa.module_name, imp)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    lines.append(f"    {fa.module_name} --> {imp}")
        lines.append("```")
        return "\n".join(lines)

    def _gen_function_flow_mermaid(self) -> str:
        lines = ["```mermaid", "flowchart TD"]

        # Collect all defined function names (for cross-module matching)
        all_defined = set()
        for fa in self.files:
            for fn in fa.functions:
                all_defined.add(fn["name"])
            for cls in fa.classes:
                for m in cls["methods"]:
                    all_defined.add(m["name"])

        # Build subgraphs per file
        for fa in self.files:
            safe_mod = fa.module_name.replace("-", "_")
            lines.append(f'    subgraph {safe_mod}["{fa.basename}"]')

            for fn in fa.functions:
                node_id = f"{safe_mod}__{fn['name']}"
                if fn["is_entry"]:
                    lines.append(f'        {node_id}[["🚀 {fn["name"]}()"]]')
                else:
                    lines.append(f'        {node_id}["{fn["name"]}()"]')

            for cls in fa.classes:
                safe_cls = cls["name"].replace("-", "_")
                lines.append(f'        subgraph {safe_mod}_{safe_cls}["{cls["name"]}"]')
                for m in cls["methods"]:
                    node_id = f'{safe_mod}__{safe_cls}__{m["name"]}'
                    lines.append(f'            {node_id}["{cls["name"]}.{m["name"]}()"]')
                lines.append("        end")

            lines.append("    end")

        # Build edges
        edges = set()
        for fa in self.files:
            safe_mod = fa.module_name.replace("-", "_")
            for fn in fa.functions:
                src_id = f"{safe_mod}__{fn['name']}"
                for call in fn["calls"]:
                    target_id = self._resolve_target_id(call, all_defined, fa)
                    if target_id and (src_id, target_id) not in edges:
                        edges.add((src_id, target_id))
                        lines.append(f"    {src_id} --> {target_id}")

            for cls in fa.classes:
                safe_cls = cls["name"].replace("-", "_")
                for m in cls["methods"]:
                    src_id = f"{safe_mod}__{safe_cls}__{m['name']}"
                    for call in m["calls"]:
                        target_id = self._resolve_target_id(call, all_defined, fa)
                        if target_id and (src_id, target_id) not in edges:
                            edges.add((src_id, target_id))
                            lines.append(f"    {src_id} --> {target_id}")

        lines.append("```")
        return "\n".join(lines)

    def _resolve_target_id(self, call_name: str, all_defined: set, source_fa: FileAnalyzer) -> str | None:
        """Resolve a function call name to its Mermaid node ID."""
        # Skip builtins and common library calls
        builtins_skip = {
            "print", "len", "range", "enumerate", "zip", "map", "filter",
            "sorted", "reversed", "list", "dict", "set", "tuple", "str",
            "int", "float", "bool", "type", "isinstance", "issubclass",
            "hasattr", "getattr", "setattr", "open", "super", "any", "all",
            "min", "max", "sum", "abs", "round", "format", "id", "hash",
            "iter", "next", "repr", "input", "vars", "dir",
        }

        # Simple name — a direct function call
        simple_name = call_name.split(".")[-1] if "." in call_name else call_name

        if simple_name in builtins_skip:
            return None

        # Check if it's a method call like self.something or ClassName.method
        parts = call_name.split(".")
        if len(parts) == 2:
            obj, method = parts
            # self.method → look in same class
            if obj == "self":
                # Find the class this method belongs to (in the source file)
                for cls in source_fa.classes:
                    for m in cls["methods"]:
                        if m["name"] == method:
                            safe_mod = source_fa.module_name.replace("-", "_")
                            safe_cls = cls["name"].replace("-", "_")
                            return f"{safe_mod}__{safe_cls}__{method}"
                return None

            # ClassName.method → look in all files
            for fa in self.files:
                safe_mod = fa.module_name.replace("-", "_")
                for cls in fa.classes:
                    if cls["name"] == obj:
                        safe_cls = cls["name"].replace("-", "_")
                        for m in cls["methods"]:
                            if m["name"] == method:
                                return f"{safe_mod}__{safe_cls}__{method}"

        # Simple function name — look in all files
        if simple_name in all_defined:
            # Prefer same file
            for fn in source_fa.functions:
                if fn["name"] == simple_name:
                    safe_mod = source_fa.module_name.replace("-", "_")
                    return f"{safe_mod}__{simple_name}"
            # Then other files
            for fa in self.files:
                safe_mod = fa.module_name.replace("-", "_")
                for fn in fa.functions:
                    if fn["name"] == simple_name:
                        return f"{safe_mod}__{simple_name}"
                # Also check class methods
                for cls in fa.classes:
                    for m in cls["methods"]:
                        if m["name"] == simple_name:
                            safe_cls = cls["name"].replace("-", "_")
                            return f"{safe_mod}__{safe_cls}__{simple_name}"

        return None


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python project code flow and generate a Mermaid flowchart report."
    )
    parser.add_argument("source_dir", help="Directory containing .py files to analyze")
    parser.add_argument("--output", default=None, help="Output Markdown file path")
    parser.add_argument("--entry", default=None, help="Entry-point file basename (e.g. main.py)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    if not os.path.isdir(source_dir):
        print(f"Error: {source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(source_dir, "code_flow_report.md")

    analyzer = ProjectAnalyzer(source_dir, recursive=args.recursive, entry_file=args.entry)
    analyzer.analyze()

    report = analyzer.generate_report()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {output_path}")
    print(f"   {len(analyzer.files)} files, {sum(len(fa.functions) for fa in analyzer.files)} functions, "
          f"{sum(len(fa.classes) for fa in analyzer.files)} classes analyzed")


if __name__ == "__main__":
    main()
