"""Unit tests for phase detection, error detection, and middleware routing."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from langchain_router import RouterMiddleware, _detect_phase, _looks_like_error
from tests.conftest import make_human, make_tool, make_tool_error


class TestLooksLikeError:
    """Error detection heuristic."""

    def test_error_keyword(self) -> None:
        """Content containing 'error' is detected."""
        assert _looks_like_error("Error: file not found")

    def test_traceback_keyword(self) -> None:
        """Content containing 'traceback' is detected."""
        assert _looks_like_error("Traceback (most recent call last):")

    def test_exception_keyword(self) -> None:
        """Content containing 'exception' is detected."""
        assert _looks_like_error("RuntimeException: connection refused")

    def test_failed_keyword(self) -> None:
        """Content containing 'failed' is detected."""
        assert _looks_like_error("Command failed with exit code 1")

    def test_normal_content(self) -> None:
        """Normal tool output is not flagged."""
        assert not _looks_like_error("def login(): pass")

    def test_empty_content(self) -> None:
        """Empty string is not an error."""
        assert not _looks_like_error("")

    def test_case_insensitive(self) -> None:
        """Detection is case insensitive."""
        assert _looks_like_error("ERROR: permission denied")
        assert _looks_like_error("TRACEBACK")


class TestDetectPhase:
    """Phase detection from message history."""

    def test_empty_messages(self) -> None:
        """Empty message list is planning phase."""
        assert _detect_phase([]) == "planning"

    def test_human_message(self) -> None:
        """Last message is HumanMessage: planning."""
        msgs: list[AnyMessage] = [make_human()]
        assert _detect_phase(msgs) == "planning"

    def test_tool_message(self) -> None:
        """Last message is successful ToolMessage: execution."""
        msgs: list[AnyMessage] = [make_human(), make_tool()]
        assert _detect_phase(msgs) == "execution"

    def test_tool_error(self) -> None:
        """Last message is error ToolMessage: recovery."""
        msgs: list[AnyMessage] = [make_human(), make_tool_error()]
        assert _detect_phase(msgs) == "recovery"

    def test_ai_message(self) -> None:
        """Last message is AIMessage: planning (model is about to act)."""
        msgs: list[AnyMessage] = [AIMessage(content="thinking")]
        assert _detect_phase(msgs) == "planning"

    def test_consecutive_tool_messages(self) -> None:
        """Multiple tool results in sequence: still execution."""
        msgs: list[AnyMessage] = [make_tool(call_id="a"), make_tool(call_id="b")]
        assert _detect_phase(msgs) == "execution"

    def test_tool_then_human(self) -> None:
        """User speaks after tool results: planning."""
        msgs: list[AnyMessage] = [make_tool(), make_human("now fix it")]
        assert _detect_phase(msgs) == "planning"

    def test_tool_status_error(self) -> None:
        """ToolMessage with status='error' is recovery regardless of content."""
        msg = ToolMessage(content="all good", tool_call_id="tc0", status="error")
        assert _detect_phase([msg]) == "recovery"

    def test_tool_status_success(self) -> None:
        """Error-like content triggers recovery even with status='success'."""
        msg = ToolMessage(
            content="Error: something broke", tool_call_id="tc0", status="success"
        )
        assert _detect_phase([msg]) == "recovery"

    def test_tool_status_success_clean(self) -> None:
        """ToolMessage with status='success' and clean content is execution."""
        msg = ToolMessage(content="file contents", tool_call_id="tc0", status="success")
        assert _detect_phase([msg]) == "execution"

    def test_tool_with_list_content(self) -> None:
        """ToolMessage with list content (multi-modal) is handled."""
        msg = ToolMessage(
            content=[{"type": "text", "text": "result"}],
            tool_call_id="tc0",
        )
        assert _detect_phase([msg]) == "execution"

    def test_tool_with_list_content_error(self) -> None:
        """ToolMessage with list content containing error is detected."""
        msg = ToolMessage(
            content=[{"type": "text", "text": "Error: failed"}],
            tool_call_id="tc0",
        )
        assert _detect_phase([msg]) == "recovery"


class TestRouterMiddleware:
    """Middleware integration surface."""

    @patch("langchain_router.init_chat_model")
    def test_execution_routes_to_fast(self, mock_init: Mock) -> None:
        """Execution phase swaps to the fast model."""
        fast_model = Mock()
        mock_init.return_value = fast_model

        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        request = Mock()
        request.messages = [make_tool()]
        override_request = Mock()
        request.override = Mock(return_value=override_request)
        handler = Mock(return_value="response")

        result = mw.wrap_model_call(request, handler)

        request.override.assert_called_once_with(model=fast_model)
        handler.assert_called_once_with(override_request)
        assert result == "response"

    @patch("langchain_router.init_chat_model")
    def test_planning_keeps_default(self, mock_init: Mock) -> None:
        """Planning phase does not override the model."""
        mock_init.return_value = Mock()

        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        request = Mock()
        request.messages = [make_human()]
        handler = Mock(return_value="response")

        result = mw.wrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    @patch("langchain_router.init_chat_model")
    def test_recovery_keeps_default(self, mock_init: Mock) -> None:
        """Recovery phase does not override the model."""
        mock_init.return_value = Mock()

        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        request = Mock()
        request.messages = [make_tool_error()]
        handler = Mock(return_value="response")

        result = mw.wrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    @patch("langchain_router.init_chat_model")
    @pytest.mark.asyncio
    async def test_async_execution_routes(self, mock_init: Mock) -> None:
        """Async handler receives fast model during execution."""
        fast_model = Mock()
        mock_init.return_value = fast_model

        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        request = Mock()
        request.messages = [make_tool()]
        override_request = Mock()
        request.override = Mock(return_value=override_request)

        async def async_handler(req: object) -> str:
            return "async_response"

        result = await mw.awrap_model_call(request, async_handler)

        request.override.assert_called_once_with(model=fast_model)
        assert result == "async_response"

    @patch("langchain_router.init_chat_model")
    @pytest.mark.asyncio
    async def test_async_planning_keeps_default(self, mock_init: Mock) -> None:
        """Async planning phase does not override the model."""
        mock_init.return_value = Mock()

        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        request = Mock()
        request.messages = [make_human()]

        async def async_handler(req: object) -> str:
            return "async_response"

        result = await mw.awrap_model_call(request, async_handler)

        request.override.assert_not_called()
        assert result == "async_response"

    @patch("langchain_router.init_chat_model")
    @pytest.mark.asyncio
    async def test_async_recovery_keeps_default(self, mock_init: Mock) -> None:
        """Async recovery phase does not override the model."""
        mock_init.return_value = Mock()

        mw = RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")

        request = Mock()
        request.messages = [make_tool_error()]

        async def async_handler(req: object) -> str:
            return "async_response"

        result = await mw.awrap_model_call(request, async_handler)

        request.override.assert_not_called()
        assert result == "async_response"

    @patch("langchain_router.init_chat_model")
    def test_handler_always_called(self, mock_init: Mock) -> None:
        """Handler is called exactly once regardless of phase."""
        mock_init.return_value = Mock()
        mw = RouterMiddleware(fast="x")

        for msgs in [[make_human()], [make_tool()], [make_tool_error()], []]:
            request = Mock()
            request.messages = msgs
            request.override = Mock(return_value=request)
            handler = Mock(return_value="ok")
            mw.wrap_model_call(request, handler)
            handler.assert_called_once()

    def test_accepts_model_object(self) -> None:
        """A pre-constructed model object can be passed directly."""
        from langchain_core.language_models.chat_models import BaseChatModel

        fast_model = Mock(spec=BaseChatModel)
        mw = RouterMiddleware(fast=fast_model)
        assert mw._fast_model is fast_model

    def test_none_fast_raises(self) -> None:
        """None raises TypeError."""
        with pytest.raises(TypeError, match="must be a model string or BaseChatModel"):
            RouterMiddleware(fast=None)

    def test_int_fast_raises(self) -> None:
        """Non-string non-model raises TypeError."""
        with pytest.raises(TypeError, match="must be a model string or BaseChatModel"):
            RouterMiddleware(fast=42)

    def test_empty_fast_raises(self) -> None:
        """Empty model identifier raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            RouterMiddleware(fast="")

    def test_whitespace_fast_raises(self) -> None:
        """Whitespace-only model identifier raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            RouterMiddleware(fast="   ")

    @patch(
        "langchain_router.init_chat_model", side_effect=Exception("unknown provider")
    )
    def test_invalid_model_raises(self, mock_init: Mock) -> None:
        """Unresolvable model identifier raises ValueError with context."""
        with pytest.raises(ValueError, match="failed to resolve fast model"):
            RouterMiddleware(fast="fake:nonexistent")

    @patch("langchain_router.init_chat_model")
    def test_composes_with_collapse_badge(self, mock_init: Mock) -> None:
        """Phase detection works correctly after CollapseMiddleware runs.

        After CollapseMiddleware collapses a group, the message list contains
        a badge ``HumanMessage`` followed by the last tool pair.  The phase
        should be determined by the final message, not the badge.
        """
        from langchain_core.messages import HumanMessage

        mock_init.return_value = Mock()
        mw = RouterMiddleware(fast="x")

        collapse_badge = HumanMessage(
            content="[4 tool results omitted]",
            additional_kwargs={"lc_source": "collapse"},
        )
        last_tool = make_tool(content="file contents", call_id="tc5")

        request = Mock()
        request.messages = [collapse_badge, last_tool]
        request.override = Mock(return_value=request)
        handler = Mock(return_value="ok")

        mw.wrap_model_call(request, handler)

        # Last message is a successful ToolMessage, so fast model is used.
        request.override.assert_called_once()

    @patch("langchain_router.init_chat_model")
    def test_real_model_request(self, mock_init: Mock) -> None:
        """Override works correctly on a real ModelRequest (not a Mock).

        Verifies that ``request.override(model=...)`` creates a new request
        with the fast model while preserving messages, tools, and state.
        """
        from langchain.agents.middleware.types import ModelRequest

        fast_model = Mock()
        mock_init.return_value = fast_model
        mw = RouterMiddleware(fast="x")

        original_model = Mock()
        request = ModelRequest(
            model=original_model,
            messages=[make_tool()],
            system_message=None,
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=Mock(),
            model_settings={},
        )

        captured: dict[str, object] = {}

        def handler(req: object) -> str:
            captured["req"] = req
            return "ok"

        mw.wrap_model_call(request, handler)

        received = captured["req"]
        assert received is not request
        assert received.model is fast_model
        assert received.messages == [make_tool()]
