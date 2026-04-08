"""Property-based invariant tests.

Verify that structural properties hold for any valid input:

1. Phase is always one of three values.
2. Handler is always called exactly once.
3. Fast model is used if and only if phase is execution.
4. Phase depends only on the last message.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AnyMessage

from langchain_router import _detect_phase
from tests.conftest import make_human, make_tool, make_tool_error

_VALID_PHASES: frozenset[str] = frozenset({"planning", "execution", "recovery"})


class TestPhaseAlwaysValid:
    """Phase detection always returns one of three values."""

    @given(n_human=st.integers(0, 10), n_tool=st.integers(0, 10))
    @settings(max_examples=50)
    def test_mixed_messages(self, n_human: int, n_tool: int) -> None:
        """Any combination of HumanMessages and ToolMessages yields a valid phase."""
        msgs: list[AnyMessage] = [make_human(f"msg {i}") for i in range(n_human)]
        msgs.extend(make_tool(call_id=f"tc{i}") for i in range(n_tool))
        assert _detect_phase(msgs) in _VALID_PHASES

    @given(n=st.integers(0, 20))
    @settings(max_examples=30)
    def test_only_errors(self, n: int) -> None:
        """A list of only error ToolMessages yields a valid phase."""
        msgs: list[AnyMessage] = [make_tool_error(call_id=f"tc{i}") for i in range(n)]
        assert _detect_phase(msgs) in _VALID_PHASES

    @given(n=st.integers(0, 20))
    @settings(max_examples=30)
    def test_only_human(self, n: int) -> None:
        """A list of only HumanMessages yields planning."""
        msgs: list[AnyMessage] = [make_human(f"msg {i}") for i in range(n)]
        assert _detect_phase(msgs) == "planning"


class TestRoutingCorrectness:
    """Fast model is used if and only if phase is execution."""

    @given(last_type=st.sampled_from(["human", "tool", "error", "empty"]))
    @settings(max_examples=40)
    @patch("langchain_router.init_chat_model")
    def test_fast_model_iff_execution(self, mock_init: Mock, last_type: str) -> None:
        """Override happens exactly when phase is execution."""
        from langchain_router import RouterMiddleware

        fast = Mock()
        mock_init.return_value = fast
        mw = RouterMiddleware(fast="x")

        msgs: list[AnyMessage]
        if last_type == "human":
            msgs = [make_human()]
        elif last_type == "tool":
            msgs = [make_tool()]
        elif last_type == "error":
            msgs = [make_tool_error()]
        else:
            msgs = []

        request = Mock()
        request.messages = msgs
        request.override = Mock(return_value=request)
        handler = Mock(return_value="ok")

        mw.wrap_model_call(request, handler)

        phase = _detect_phase(msgs)
        if phase == "execution":
            request.override.assert_called_once_with(model=fast)
        else:
            request.override.assert_not_called()


class TestPhaseDependsOnlyOnLast:
    """Phase is determined solely by the last message."""

    @given(
        n_prefix=st.integers(0, 10),
        last_type=st.sampled_from(["human", "tool", "error"]),
    )
    @settings(max_examples=50)
    def test_prefix_irrelevant(self, n_prefix: int, last_type: str) -> None:
        """Changing the prefix does not change the phase."""
        if last_type == "human":
            last = make_human()
        elif last_type == "tool":
            last = make_tool()
        else:
            last = make_tool_error()

        # Phase with just the last message.
        expected = _detect_phase([last])

        # Phase with a random prefix + same last message.
        prefix: list[AnyMessage] = [make_human(f"p{i}") for i in range(n_prefix)]
        assert _detect_phase([*prefix, last]) == expected


class TestHandlerAlwaysCalled:
    """The handler is called exactly once regardless of phase."""

    @given(last_type=st.sampled_from(["human", "tool", "error", "empty"]))
    @settings(max_examples=30)
    @patch("langchain_router.init_chat_model")
    def test_handler_called_once(self, mock_init: Mock, last_type: str) -> None:
        """Handler is invoked exactly once for any message type."""
        from langchain_router import RouterMiddleware

        mock_init.return_value = Mock()
        mw = RouterMiddleware(fast="x")

        msgs: list[AnyMessage]
        if last_type == "human":
            msgs = [make_human()]
        elif last_type == "tool":
            msgs = [make_tool()]
        elif last_type == "error":
            msgs = [make_tool_error()]
        else:
            msgs = []

        request = Mock()
        request.messages = msgs
        request.override = Mock(return_value=request)
        handler = Mock(return_value="ok")

        mw.wrap_model_call(request, handler)
        handler.assert_called_once()
