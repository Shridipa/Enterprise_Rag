from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".md": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
}

CPP_CLASS_PATTERN = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([^\{]+))?\s*\{", re.M)
CPP_NAMESPACE_PATTERN = re.compile(r"\bnamespace\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
CPP_FUNCTION_PATTERN = re.compile(r"(?:^|\n)\s*(?:[A-Za-z_:<>*&\s]+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*(?:const\s*)?(?:\{|;)", re.M)
CPP_INCLUDE_PATTERN = re.compile(r"#include\s+[<\"]([^>\"]+)[>\"]")


@dataclass
class RepositoryIndex:
    repo_id: str
    name: str
    path: str
    summary: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)


class RepositoryIndexer:
    def __init__(self, repo_root: str | os.PathLike[str]):
        self.repo_root = Path(repo_root).resolve()

    def index(self) -> RepositoryIndex:
        file_entries: list[dict[str, Any]] = []
        for path in sorted(self.repo_root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            content = path.read_text(encoding="utf-8", errors="ignore")
            rel_path = path.relative_to(self.repo_root).as_posix()
            language = SUPPORTED_EXTENSIONS[suffix]
            entry = {
                "path": rel_path,
                "language": language,
                "size": len(content.encode("utf-8")),
                "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "analysis": self._analyze_source_file(rel_path, content, language),
            }
            if suffix == ".py":
                entry["symbols"] = self._extract_python_symbols(content)
            file_entries.append(entry)

        summary = self._build_summary(file_entries)
        graph = self._build_graph(file_entries)
        repo_id = hashlib.sha256(str(self.repo_root).encode("utf-8")).hexdigest()[:12]
        return RepositoryIndex(
            repo_id=repo_id,
            name=self.repo_root.name,
            path=str(self.repo_root),
            summary=summary,
            files=file_entries,
            graph=graph,
        )

    def _build_summary(self, file_entries: list[dict[str, Any]]) -> dict[str, Any]:
        languages = sorted({entry["language"] for entry in file_entries})
        analysis: dict[str, Any] = {}
        for entry in file_entries:
            language = entry["language"]
            analysis.setdefault(language, {
                "file_count": 0,
                "function_count": 0,
                "class_count": 0,
                "namespace_count": 0,
                "symbol_count": 0,
            })
            analysis[language]["file_count"] += 1
            file_analysis = entry.get("analysis", {})
            analysis[language]["function_count"] += int(file_analysis.get("function_count", 0))
            analysis[language]["class_count"] += int(file_analysis.get("class_count", 0))
            analysis[language]["namespace_count"] += int(file_analysis.get("namespace_count", 0))
            analysis[language]["symbol_count"] += len(file_analysis.get("symbols", []))

        return {
            "file_count": len(file_entries),
            "languages": languages,
            "largest_files": sorted(file_entries, key=lambda item: item["size"], reverse=True)[:5],
            "analysis": analysis,
        }

    def _analyze_source_file(self, rel_path: str, content: str, language: str) -> dict[str, Any]:
        if language == "python":
            symbols = self._extract_python_symbols(content)
            return {"symbols": symbols, "function_count": len([name for name in symbols if not name.startswith("_")]), "class_count": 0, "namespace_count": 0}
        if language == "cpp":
            return self._analyze_cpp_content(content, rel_path)
        return {"symbols": [], "function_count": 0, "class_count": 0, "namespace_count": 0}

    def _analyze_cpp_content(self, content: str, rel_path: str) -> dict[str, Any]:
        classes = [match.group(1) for match in CPP_CLASS_PATTERN.finditer(content)]
        namespaces = [match.group(1) for match in CPP_NAMESPACE_PATTERN.finditer(content)]
        functions = [match.group(1) for match in CPP_FUNCTION_PATTERN.finditer(content)]
        symbols = classes + namespaces + functions
        regex_result = {
            "symbols": symbols,
            "function_count": len(functions),
            "class_count": len(classes),
            "namespace_count": len(namespaces),
            "imports": [match.group(1) for match in CPP_INCLUDE_PATTERN.finditer(content)],
            "source": "regex-fallback",
            "path": rel_path,
        }

        if regex_result["class_count"] or regex_result["function_count"] or regex_result["namespace_count"]:
            return regex_result

        clang_result = self._analyze_cpp_with_clang(content)
        if clang_result is not None:
            return clang_result
        return regex_result

    def _analyze_cpp_with_clang(self, content: str) -> dict[str, Any] | None:
        try:
            proc = subprocess.run(
                ["clang++", "-x", "c++", "-fsyntax-only", "-Xclang", "-ast-dump", "-"],
                input=content,
                text=True,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        stdout = proc.stdout or ""
        class_count = len(re.findall(r"CXXRecordDecl", stdout))
        namespace_count = len(re.findall(r"NamespaceDecl", stdout))
        function_count = len(re.findall(r"CXXMethodDecl|FunctionDecl", stdout))
        return {
            "symbols": [],
            "function_count": function_count,
            "class_count": class_count,
            "namespace_count": namespace_count,
            "source": "clang-ast",
        }

    def _extract_python_symbols(self, content: str) -> list[str]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        symbols: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        return symbols

    def _build_graph(self, file_entries: list[dict[str, Any]]) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for entry in file_entries:
            nodes.append({"id": entry["path"], "type": "file", "language": entry["language"], "path": entry["path"]})

        for entry in file_entries:
            content = (self.repo_root / entry["path"]).read_text(encoding="utf-8", errors="ignore")
            if entry["language"] == "python":
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if module:
                            edges.append({"source": entry["path"], "target": f"import:{module}", "type": "import"})
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            edges.append({"source": entry["path"], "target": f"import:{alias.name}", "type": "import"})
            elif entry["language"] == "cpp":
                for class_name, base_list in self._extract_cpp_inheritance(content):
                    for base_name in base_list:
                        edges.append({
                            "source": f"{entry['path']}:{class_name}",
                            "target": f"{entry['path']}:{base_name}",
                            "type": "inheritance",
                        })
                for include_name in re.findall(r"#include\s+[<\"]([^>\"]+)[>\"]", content):
                    edges.append({"source": entry["path"], "target": f"include:{include_name}", "type": "include"})
        return {"nodes": nodes, "edges": edges}

    def _extract_cpp_inheritance(self, content: str) -> list[tuple[str, list[str]]]:
        results: list[tuple[str, list[str]]] = []
        for match in CPP_CLASS_PATTERN.finditer(content):
            class_name = match.group(1)
            bases = [piece.strip() for piece in (match.group(2) or "").split(",") if piece.strip()]
            cleaned_bases: list[str] = []
            for base in bases:
                base_name = re.sub(r"\b(?:public|private|protected|virtual|struct)\b", "", base).strip()
                base_name = base_name.replace("::", "").strip()
                if base_name:
                    cleaned_bases.append(base_name)
            if cleaned_bases:
                results.append((class_name, cleaned_bases))
        return results


def index_repository(path: str | os.PathLike[str]) -> RepositoryIndex:
    return RepositoryIndexer(path).index()


def serialize_index(index: RepositoryIndex) -> str:
    return json.dumps(index.__dict__, default=str)
