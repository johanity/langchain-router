"""Full verification: 9 model pairs × 9 calls = 81 real API calls.

Reproduces the results in full_verification.md. Requires API keys
for OpenRouter (OPENROUTER_API_KEY) and OpenAI (OPENAI_API_KEY).

Usage::

    OPENROUTER_API_KEY=... python examples/run_full_verification.py
"""

import signal
import time

from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_router import RouterMiddleware

# Per-call timeout (seconds). Increase for slow providers.
CALL_TIMEOUT = 300


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return (
        "def login(user, pw):\n"
        "    if not verify(pw):\n"
        "        raise AuthError('bad')\n"
        "    return session(user)"
    )


def _make_request(model, messages):
    """Build a ModelRequest with the given model and messages."""
    return ModelRequest(
        model=model,
        messages=messages,
        system_message=None,
        tool_choice=None,
        tools=[read_file],
        response_format=None,
        state={"messages": messages},
        runtime=type("R", (), {
            "context": None,
            "stream_writer": None,
            "store": None,
            "config": {},
        })(),
        model_settings={},
    )


def _session_calls():
    """Return 9 calls simulating a full agent session.

    Covers: planning, 6 execution turns, recovery, and re-planning.
    """
    # Simulates a real agent session debugging a LangChain project.
    # Planning: user describes the task.
    # Execution: agent reads source files (tool calls succeed).
    # Recovery: agent hits an ImportError (tool output contains traceback).
    # Re-planning: user adds a follow-up request.
    return [
        ("plan", [HumanMessage(content="BaseTool.arun handles pydantic v1 validation differently from run, can you investigate?")], False),
        ("exec1", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "langchain_core/tools/base.py"}, "id": "t1"}]), ToolMessage(content="class BaseTool(RunnableSerializable):\n    name: str\n    description: str\n    args_schema: type[BaseModel] | None = None\n    def _run(self, *args, **kwargs): ...\n    async def _arun(self, *args, **kwargs): ...", tool_call_id="t1")], True),
        ("exec2", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "langchain_core/tools/structured.py"}, "id": "t2"}]), ToolMessage(content="class StructuredTool(BaseTool):\n    func: Callable[..., str] | None = None\n    coroutine: Callable[..., Awaitable[str]] | None = None\n    def _run(self, *args, **kwargs):\n        return self.func(*args, **kwargs)", tool_call_id="t2")], True),
        ("exec3", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "langchain_core/runnables/base.py"}, "id": "t3"}]), ToolMessage(content="class RunnableSerializable(Serializable, Runnable):\n    def invoke(self, input, config=None, **kwargs):\n        return self._call_with_config(self._invoke, input, config, **kwargs)", tool_call_id="t3")], True),
        ("exec4", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "langchain/agents/middleware/types.py"}, "id": "t4"}]), ToolMessage(content="class AgentMiddleware(Generic[StateT_co, ContextT]):\n    state_schema: type[AgentState] = AgentState\n    def wrap_model_call(self, request, handler): return handler(request)\n    async def awrap_model_call(self, request, handler): return await handler(request)", tool_call_id="t4")], True),
        ("exec5", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "tests/unit_tests/test_tools.py"}, "id": "t5"}]), ToolMessage(content="def test_tool_arun():\n    tool = StructuredTool.from_function(func=lambda x: x, name='t', description='d')\n    result = asyncio.run(tool.arun({'x': 'hello'}))\n    assert result == 'hello'", tool_call_id="t5")], True),
        ("recov", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "run_tests.py"}, "id": "t6"}]), ToolMessage(content="Error: FAILED tests/unit_tests/test_tools.py::test_tool_arun\nTraceback (most recent call last):\n  File \"tests/unit_tests/test_tools.py\", line 4, in test_tool_arun\n    result = asyncio.run(tool.arun({'x': 'hello'}))\nValidationError: 1 validation error for StructuredTool\n  value is not a valid dict", tool_call_id="t6")], False),
        ("exec6", [HumanMessage(content="go"), AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "langchain_core/tools/base.py"}, "id": "t7"}]), ToolMessage(content="File edited successfully", tool_call_id="t7")], True),
        ("plan2", [HumanMessage(content="Also check if filter_messages leaves orphaned function_call blocks after excluding tool calls")], False),
    ]


PAIRS = [
    ("GLM-5 → MiniMax M2.7", "openrouter:z-ai/glm-5", "openrouter:minimax/minimax-m2.7"),
    ("GPT-5.4 → GPT-4.1-mini", "openai:gpt-5.4", "openai:gpt-4.1-mini"),
    ("Sonnet → GLM-5", "openrouter:anthropic/claude-sonnet-4-6", "openrouter:z-ai/glm-5"),
    ("Sonnet → MiniMax M2.7", "openrouter:anthropic/claude-sonnet-4-6", "openrouter:minimax/minimax-m2.7"),
    ("GPT-5.4 → GLM-5", "openai:gpt-5.4", "openrouter:z-ai/glm-5"),
    ("GPT-5.4 → MiniMax M2.7", "openai:gpt-5.4", "openrouter:minimax/minimax-m2.7"),
    ("Opus → Haiku", "openrouter:anthropic/claude-opus-4-6", "openrouter:anthropic/claude-haiku-4-5"),
    ("Opus → GLM-5", "openrouter:anthropic/claude-opus-4-6", "openrouter:z-ai/glm-5"),
    ("Opus → MiniMax M2.7", "openrouter:anthropic/claude-opus-4-6", "openrouter:minimax/minimax-m2.7"),
]


def _on_timeout(signum, frame):
    raise TimeoutError("API call exceeded timeout")


def main():
    print(f"# Full Verification: {len(PAIRS) * 9} Real API Calls")
    print()
    print(f"{len(PAIRS)} model pairs × 9 calls each.")
    print()
    print(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    total_correct = 0
    total_calls = 0
    all_results = []

    for pair_name, primary_str, fast_str in PAIRS:
        print(f"## {pair_name}")
        print()
        try:
            primary = init_chat_model(primary_str)
            mw = RouterMiddleware(fast=fast_str)
        except Exception as exc:
            print(f"Setup failed: {exc}")
            print()
            all_results.append((pair_name, 0, 9))
            continue

        pair_correct = 0
        print("| Call | Expected | Got | Status |")
        print("|------|----------|-----|--------|")

        for label, msgs, expect_fast in _session_calls():
            total_calls += 1
            req = _make_request(primary, msgs)
            captured = {}

            def handler(r, _cap=captured):
                _cap["model"] = r.model
                _cap["response"] = r.model.invoke(r.messages)
                return "ok"

            try:
                signal.signal(signal.SIGALRM, _on_timeout)
                signal.alarm(CALL_TIMEOUT)
                mw.wrap_model_call(req, handler)
                signal.alarm(0)

                used_fast = captured["model"] is mw._fast_model
                correct = used_fast == expect_fast
                if correct:
                    pair_correct += 1
                    total_correct += 1

                got = "fast" if used_fast else "primary"
                expected = "fast" if expect_fast else "primary"
                status = "✓" if correct else "✗"
                print(f"| {label} | {expected} | {got} | {status} |")
            except Exception as exc:
                expected = "fast" if expect_fast else "primary"
                print(f"| {label} | {expected} | error | ✗ ({exc!s:.30}) |")

        print()
        print(f"**{pair_correct}/9 correct**")
        print()
        all_results.append((pair_name, pair_correct, 9))

    print("## Summary")
    print()
    print("| Pair | Result |")
    print("|------|--------|")
    for name, correct, total in all_results:
        print(f"| {name} | {correct}/{total} |")
    print()
    print(f"**Total: {total_correct}/{total_calls} correct across {len(PAIRS)} model pairs**")


if __name__ == "__main__":
    main()
