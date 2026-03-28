"""
Tool-Using Agents
Agents can CALL TOOLS mid-debate: web search, code execution,
calculator, Wikipedia. Evidence-grounded reasoning vs pure opinion exchange.
"""

from __future__ import annotations
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"tool": self.tool_name, "success": self.success, "output": self.output[:500]}


class Tool:
    """Base class for all agent tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, input_text: str) -> ToolResult:
        raise NotImplementedError


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__("calculator", "Evaluate mathematical expressions safely.")

    def run(self, expression: str) -> ToolResult:
        try:
            # Safe eval: only allow math operations
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed.update({"abs": abs, "round": round, "int": int, "float": float})
            result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
            return ToolResult(self.name, True, str(result))
        except Exception as e:
            return ToolResult(self.name, False, "", str(e))


class WikipediaTool(Tool):
    def __init__(self):
        super().__init__("wikipedia", "Fetch a Wikipedia article summary.")

    def run(self, query: str) -> ToolResult:
        try:
            encoded = urllib.parse.quote(query.replace(" ", "_"))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "CognitiveConflictAI/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                summary = data.get("extract", "No summary found.")[:800]
                return ToolResult(self.name, True, summary)
        except Exception as e:
            return ToolResult(self.name, False, "", str(e))


class WebSearchTool(Tool):
    """
    Lightweight web search using DuckDuckGo Instant Answer API (no key required).
    For production, replace with SerpAPI, Brave Search, or Bing.
    """

    def __init__(self):
        super().__init__("web_search", "Search the web for current information.")

    def run(self, query: str) -> ToolResult:
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1"
            req = urllib.request.Request(url, headers={"User-Agent": "CognitiveConflictAI/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                abstract = data.get("AbstractText", "")
                answer = data.get("Answer", "")
                result = abstract or answer or "No instant answer found."
                return ToolResult(self.name, True, result[:600])
        except Exception as e:
            return ToolResult(self.name, False, "", str(e))


class PythonCodeTool(Tool):
    """
    Executes simple Python expressions in a sandboxed environment.
    WARNING: In production, use a proper sandbox (Firecracker, gVisor, etc.)
    """

    ALLOWED_IMPORTS = {"math", "json", "re", "datetime", "collections", "itertools"}

    def __init__(self):
        super().__init__("python", "Execute simple Python code.")

    def run(self, code: str) -> ToolResult:
        if len(code) > 500:
            return ToolResult(self.name, False, "", "Code too long (max 500 chars).")
        try:
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            exec(code, {"__builtins__": {"print": print, "len": len, "range": range,
                                          "list": list, "dict": dict, "str": str,
                                          "int": int, "float": float, "sum": sum,
                                          "max": max, "min": min, "sorted": sorted}})
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            return ToolResult(self.name, True, output or "(no output)")
        except Exception as e:
            sys.stdout = old_stdout
            return ToolResult(self.name, False, "", str(e))


class ToolRegistry:
    """Central registry for all available agent tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def run(self, tool_name: str, input_text: str) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(tool_name, False, "", f"Tool '{tool_name}' not found.")
        return tool.run(input_text)

    def available(self) -> list[str]:
        return list(self._tools.keys())


def default_registry() -> ToolRegistry:
    return (
        ToolRegistry()
        .register(CalculatorTool())
        .register(WikipediaTool())
        .register(WebSearchTool())
        .register(PythonCodeTool())
    )


class ToolUseParser:
    """
    Parses tool use instructions from agent LLM output.
    Expected format in agent response:
      TOOL_USE: calculator | 2 ** 10 + sqrt(25)
      TOOL_USE: wikipedia | quantum entanglement
    """

    PREFIX = "TOOL_USE:"

    @classmethod
    def extract(cls, text: str) -> list[tuple[str, str]]:
        calls = []
        for line in text.splitlines():
            if line.startswith(cls.PREFIX):
                rest = line[len(cls.PREFIX):].strip()
                if "|" in rest:
                    tool_name, tool_input = rest.split("|", 1)
                    calls.append((tool_name.strip(), tool_input.strip()))
        return calls
