"""Model-agnostic LLM dispatcher.

Hides provider differences (Anthropic vs Google) behind a single
make_llm() factory and small helpers for extracting tool calls and
token usage from AIMessage objects.
"""

from __future__ import annotations

import os
from typing import Any

import dotenv


def _load_env_once() -> None:
    repo_env = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
    if os.path.exists(repo_env):
        dotenv.load_dotenv(repo_env, override=False)


def make_llm(provider: str, model_id: str, temperature: float = 0.1) -> Any:
    """Return a tool-callable chat model. Caller must .bind_tools() separately."""
    _load_env_once()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider!r}")


def extract_tool_calls(ai_msg: Any) -> list[dict]:
    """Return the list of tool-call dicts from an AIMessage.

    Both providers expose .tool_calls in langchain >= 0.2, but the
    shape can vary. This wrapper normalises to dicts of {name, args, id}.
    """
    calls = getattr(ai_msg, "tool_calls", None) or []
    out = []
    for tc in calls:
        if isinstance(tc, dict):
            out.append({
                "name": tc.get("name"),
                "args": tc.get("args", {}),
                "id": tc.get("id", ""),
            })
        else:
            out.append({
                "name": getattr(tc, "name", None),
                "args": getattr(tc, "args", {}),
                "id": getattr(tc, "id", ""),
            })
    return out


def extract_token_usage(ai_msg: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens). Best-effort; zeros if unknown."""
    usage = getattr(ai_msg, "usage_metadata", None)
    if usage:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    meta = getattr(ai_msg, "response_metadata", {}) or {}
    u = meta.get("usage") or meta.get("token_usage") or {}
    return (
        int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0),
        int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0),
    )


# Convenience model-id shortcuts used by notebooks
KNOWN_MODELS = {
    "haiku":  ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet": ("anthropic", "claude-sonnet-4-6-20250929"),
    "gemini": ("google",    "gemini-2.5-flash"),
}


def short_model_name(provider: str, model_id: str) -> str:
    """Compact identifier for run IDs / filenames."""
    if "haiku" in model_id:
        return "haiku"
    if "sonnet" in model_id:
        return "sonnet"
    if "gemini" in model_id:
        return "gemini"
    return model_id.replace("/", "-").replace(":", "-")
