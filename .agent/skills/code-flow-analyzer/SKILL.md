---
name: code-flow-analyzer
description: Analyze the program flow of a Python project and generate a Mermaid flowchart showing file relationships and function call chains. Use this skill whenever the user asks to analyze, visualize, or document the code flow, program structure, function call graph, or execution pipeline of a project. Also trigger when the user mentions terms like "程式流程", "code flow", "call graph", "function relationships", or asks "how does this code work" for an entire directory of Python files.
---

# Code Flow Analyzer

Analyze Python source files in a project directory, trace function call chains, and generate a Mermaid flowchart with a comprehensive Markdown report.

## When to Use

- User asks to analyze or visualize the program flow of a project
- User wants to understand how Python files and functions connect
- User says "幫我分析這個專案的程式流程" or similar
- User wants a call graph or execution pipeline diagram

## How It Works

Run the bundled analysis script on a source directory:

```bash
python <skill-path>/scripts/analyze_flow.py <source_directory> [--output <output_path>] [--entry <entry_file>]
```

**Arguments:**
- `<source_directory>` — Path to the directory containing `.py` files to analyze (required)
- `--output` — Output path for the Markdown report (default: `<source_directory>/code_flow_report.md`)
- `--entry` — Entry-point file basename (e.g. `main.py`) to highlight in the diagram. If not specified, the script auto-detects files with `if __name__ == "__main__"` blocks.

**Example:**
```bash
python .agent/skills/code-flow-analyzer/scripts/analyze_flow.py ./src
```

## Output

The script generates a Markdown file containing:

1. **Project Overview** — List of analyzed files with brief descriptions
2. **Module Dependency Graph** — Mermaid diagram showing import relationships between files
3. **Function Call Flow** — Mermaid flowchart showing function-level call chains, grouped by file (subgraph)
4. **File Details** — Per-file breakdown of classes, functions, and their call targets

## Important Notes

- The analysis uses Python's `ast` module for static analysis — it traces function definitions and call sites at the source level, not at runtime.
- Only `.py` files in the specified directory are analyzed (non-recursive by default). Use `--recursive` to include subdirectories.
- Cross-module function calls are resolved by matching function names to definitions found across all analyzed files.
- The Mermaid diagram uses subgraphs to group functions by file, making it easy to see which module each function belongs to.
