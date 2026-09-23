"""
Stage 7 – Local / fake tools

- Deterministic latency for testing
- Async execution with timeout
- Never block the conversational path longer than necessary
- Tool latency measured separately from TTFA
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[Any]]
    timeout_ms: int = 800
    idempotent: bool = True  # R7: safe to retry / duplicate


# --- Fake vehicle tools (deterministic) ---

async def _get_vehicle_status() -> dict:
    await asyncio.sleep(0.045)  # simulated 45 ms
    return {
        "location": "Downtown Garage, Level 2",
        "locked": True,
        "odometer_km": 14280,
        "alerts": [],
    }


async def _get_battery() -> dict:
    await asyncio.sleep(0.030)
    return {
        "percent": 82,
        "range_km": 310,
        "charging": False,
        "plugged_in": False,
    }


async def _get_climate() -> dict:
    await asyncio.sleep(0.025)
    return {
        "cabin_c": 22.5,
        "target_c": 21.0,
        "ac_on": False,
    }


async def _honk_flash() -> dict:
    await asyncio.sleep(0.080)
    return {"action": "honk_and_flash", "status": "ok"}


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "get_vehicle_status": ToolSpec(
        name="get_vehicle_status",
        description="Get current vehicle location, lock state, odometer, and alerts.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_get_vehicle_status,
        timeout_ms=500,
        idempotent=True,
    ),
    "get_battery": ToolSpec(
        name="get_battery",
        description="Get battery percent, estimated range, and charging state.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_get_battery,
        timeout_ms=400,
        idempotent=True,
    ),
    "get_climate": ToolSpec(
        name="get_climate",
        description="Get cabin temperature and climate control state.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_get_climate,
        timeout_ms=400,
        idempotent=True,
    ),
    "honk_flash": ToolSpec(
        name="honk_flash",
        description="Honk the horn and flash the lights to help locate the vehicle.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_honk_flash,
        timeout_ms=600,
        # Physically not idempotent; mock still allows retry. Production should
        # dedupe by tool_call_id for a short window.
        idempotent=False,
    ),
}


def list_tools_for_llm() -> list[dict]:
    """OpenAI-style tool schemas for the LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in TOOL_REGISTRY.values()
    ]


# Short-window dedupe for non-idempotent tools (tool_call_id → result)
_RECENT_CALLS: dict[str, ToolResult] = {}
_RECENT_MAX = 64


async def run_tool(
    name: str,
    args: Optional[dict] = None,
    tool_call_id: Optional[str] = None,
) -> ToolResult:
    """Execute a tool with timeout. Returns ToolResult with latency_ms."""
    spec = TOOL_REGISTRY.get(name)
    if not spec:
        return ToolResult(name=name, ok=False, error=f"Unknown tool: {name}")

    # Dedupe non-idempotent tools by tool_call_id
    if tool_call_id and not spec.idempotent and tool_call_id in _RECENT_CALLS:
        cached = _RECENT_CALLS[tool_call_id]
        return ToolResult(
            name=cached.name,
            ok=cached.ok,
            data=cached.data,
            error=cached.error,
            latency_ms=0.0,
        )

    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            spec.handler(**(args or {})),
            timeout=spec.timeout_ms / 1000.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        tr = ToolResult(name=name, ok=True, data=result, latency_ms=latency)
        if tool_call_id and not spec.idempotent:
            _RECENT_CALLS[tool_call_id] = tr
            if len(_RECENT_CALLS) > _RECENT_MAX:
                # drop oldest arbitrary keys
                for k in list(_RECENT_CALLS.keys())[: len(_RECENT_CALLS) - _RECENT_MAX]:
                    _RECENT_CALLS.pop(k, None)
        return tr
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult(name=name, ok=False, error="timeout", latency_ms=latency)
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult(name=name, ok=False, error=str(e), latency_ms=latency)
