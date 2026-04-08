"""Integration tests with real LangChain objects.

Tests the full middleware pipeline using real ``ModelRequest``,
real message types, and real ``CollapseMiddleware`` composition.
The only mocks are the model instances (no API keys needed).
"""

from __future__ import annotations

import importlib.util
from unittest.mock import Mock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from langchain_router import RouterMiddleware

_has_collapse = importlib.util.find_spec("langchain_collapse") is not None
_has_anthropic = importlib.util.find_spec("langchain_anthropic") is not None
_has_openai = importlib.util.find_spec("langchain_openai") is not None


def _build_read_session(n_files: int) -> list[AnyMessage]:
    """Build a session where the agent reads *n_files* consecutively."""
    msgs: list[AnyMessage] = [HumanMessage(content="Fix the bug")]
    for i in range(n_files):
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": f"/src/f{i}.py"},
                        "id": f"tc{i}",
                    }
                ],
            ),
        )
        msgs.append(
            ToolMessage(content=f"# file {i}\ndef main(): pass", tool_call_id=f"tc{i}")
        )
    return msgs


def _make_request(model: object, messages: list[AnyMessage]) -> ModelRequest:
    """Construct a real ``ModelRequest``."""
    return ModelRequest(
        model=model,
        messages=messages,
        system_message=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": messages},
        runtime=Mock(),
        model_settings={},
    )


class TestRealModelRequest:
    """Verify middleware behavior with real framework objects."""

    @patch("langchain_router.init_chat_model")
    def test_execution_overrides_real_request(self, mock_init: Mock) -> None:
        """Override produces a new ModelRequest with fast model and preserved fields."""
        primary = Mock(name="sonnet")
        fast = Mock(name="haiku")
        mock_init.return_value = fast

        mw = RouterMiddleware(fast="x")
        msgs = [ToolMessage(content="file content", tool_call_id="tc0")]
        request = _make_request(primary, msgs)

        captured: dict[str, object] = {}

        def handler(req: object) -> str:
            captured["req"] = req
            return "ok"

        mw.wrap_model_call(request, handler)

        received = captured["req"]
        assert received is not request
        assert received.model is fast
        assert received.messages == msgs

    @patch("langchain_router.init_chat_model")
    def test_planning_preserves_real_request(self, mock_init: Mock) -> None:
        """Planning phase passes the original request unchanged."""
        primary = Mock(name="sonnet")
        mock_init.return_value = Mock()

        mw = RouterMiddleware(fast="x")
        msgs = [HumanMessage(content="hello")]
        request = _make_request(primary, msgs)

        captured: dict[str, object] = {}

        def handler(req: object) -> str:
            captured["req"] = req
            return "ok"

        mw.wrap_model_call(request, handler)

        assert captured["req"] is request
        assert captured["req"].model is primary


@pytest.mark.skipif(not _has_collapse, reason="langchain-collapse not installed")
class TestCollapseComposition:
    """Verify CollapseMiddleware and RouterMiddleware compose correctly."""

    @patch("langchain_router.init_chat_model")
    def test_full_pipeline(self, mock_init: Mock) -> None:
        """Collapse reduces messages, router routes to fast model.

        Pipeline: 13 messages -> CollapseMiddleware -> 4 messages
        -> RouterMiddleware -> fast model (execution phase).
        """
        from langchain_collapse import CollapseMiddleware

        primary = Mock(name="sonnet")
        fast = Mock(name="haiku")
        mock_init.return_value = fast

        collapse_mw = CollapseMiddleware()
        router_mw = RouterMiddleware(fast="x")

        msgs = _build_read_session(6)
        request = _make_request(primary, msgs)

        captured: dict[str, object] = {}

        def model_call(req: object) -> str:
            captured["req"] = req
            return "response"

        def router_handler(req: ModelRequest) -> str:
            return router_mw.wrap_model_call(req, model_call)

        result = collapse_mw.wrap_model_call(request, router_handler)

        final = captured["req"]

        # Collapse reduced messages.
        assert len(final.messages) < len(msgs)

        # Router selected fast model (last message is successful ToolMessage).
        assert final.model is fast

        # Badge is present with correct metadata.
        badges = [
            m
            for m in final.messages
            if isinstance(m, HumanMessage) and "omitted" in str(m.content)
        ]
        assert len(badges) == 1
        assert badges[0].additional_kwargs.get("lc_source") == "collapse"

        assert result == "response"

    @patch("langchain_router.init_chat_model")
    def test_pipeline_recovery_after_collapse(self, mock_init: Mock) -> None:
        """After collapse, an error routes to the primary model."""
        from langchain_collapse import CollapseMiddleware

        primary = Mock(name="sonnet")
        fast = Mock(name="haiku")
        mock_init.return_value = fast

        collapse_mw = CollapseMiddleware()
        router_mw = RouterMiddleware(fast="x")

        msgs = _build_read_session(4)
        # Add an error after the reads.
        msgs.append(
            AIMessage(
                content="", tool_calls=[{"name": "execute", "args": {}, "id": "err"}]
            ),
        )
        msgs.append(
            ToolMessage(
                content="Error: FAILED\nTraceback: AssertionError", tool_call_id="err"
            ),
        )

        request = _make_request(primary, msgs)

        captured: dict[str, object] = {}

        def model_call(req: object) -> str:
            captured["req"] = req
            return "recovery"

        def router_handler(req: ModelRequest) -> str:
            return router_mw.wrap_model_call(req, model_call)

        collapse_mw.wrap_model_call(request, router_handler)

        # Last message is an error ToolMessage: recovery phase, primary model.
        assert captured["req"].model is primary


@pytest.mark.skipif(not _has_anthropic, reason="langchain-anthropic not installed")
class TestRealAnthropicModels:
    """Route between real ChatAnthropic model objects."""

    def test_sonnet_to_haiku(self) -> None:
        """Resolve real Sonnet and Haiku, verify routing with real objects."""
        import os

        from langchain.chat_models import init_chat_model

        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-dummy")

        primary = init_chat_model("anthropic:claude-sonnet-4-6")
        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        for msgs, expect_fast in [
            ([HumanMessage(content="plan")], False),
            ([ToolMessage(content="file content", tool_call_id="tc0")], True),
            ([ToolMessage(content="Error: crash", tool_call_id="tc0")], False),
        ]:
            request = _make_request(primary, msgs)
            captured: dict[str, object] = {}

            def handler(req: object, _c: dict = captured) -> str:
                _c["model"] = req.model
                return "ok"

            mw.wrap_model_call(request, handler)

            used_fast = captured["model"] is mw._fast_model
            assert used_fast == expect_fast


@pytest.mark.skipif(not _has_openai, reason="langchain-openai not installed")
class TestRealOpenAIModels:
    """Route between real ChatOpenAI model objects."""

    def test_gpt5_to_mini(self) -> None:
        """Resolve real GPT-5.4 and GPT-4.1-mini, verify routing."""
        import os

        from langchain.chat_models import init_chat_model

        os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")

        primary = init_chat_model("openai:gpt-5.4")
        mw = RouterMiddleware(fast="openai:gpt-4.1-mini")

        for msgs, expect_fast in [
            ([HumanMessage(content="plan")], False),
            ([ToolMessage(content="file content", tool_call_id="tc0")], True),
            ([ToolMessage(content="Error: crash", tool_call_id="tc0")], False),
        ]:
            request = _make_request(primary, msgs)
            captured: dict[str, object] = {}

            def handler(req: object, _c: dict = captured) -> str:
                _c["model"] = req.model
                return "ok"

            mw.wrap_model_call(request, handler)

            used_fast = captured["model"] is mw._fast_model
            assert used_fast == expect_fast
