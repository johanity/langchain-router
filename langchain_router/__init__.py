"""Phase-based model routing middleware for LangChain agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

__all__ = ["RouterMiddleware"]

_ERROR_MARKERS: frozenset[str] = frozenset(
    {
        "error",
        "traceback",
        "exception",
        "failed",
    }
)


def _looks_like_error(content: str) -> bool:
    """Check whether tool output looks like an error.

    Args:
        content: Tool result content string.

    Returns:
        True if common error markers are present.
    """
    lowered = content.lower()
    return any(marker in lowered for marker in _ERROR_MARKERS)


def _detect_phase(messages: list[AnyMessage]) -> str:
    """Detect the current agent phase from the message history.

    Phases:

    - ``planning``: the model needs to reason about the task (user just
      spoke, conversation is new, or no messages yet).
    - ``execution``: the model is selecting the next tool call
      (last message is a successful tool result).
    - ``recovery``: a tool call failed and the model needs to reason
      about what went wrong.

    Args:
        messages: Current message list.

    Returns:
        One of ``"planning"``, ``"execution"``, or ``"recovery"``.
    """
    if not messages:
        return "planning"

    last = messages[-1]

    if isinstance(last, ToolMessage):
        if getattr(last, "status", None) == "error":
            return "recovery"
        content = last.content if isinstance(last.content, str) else str(last.content)
        if _looks_like_error(content):
            return "recovery"
        return "execution"

    return "planning"


class RouterMiddleware(AgentMiddleware):
    """Route model calls to a fast model during execution phases.

    Detects whether the agent is planning, executing, or recovering and
    swaps to a fast model during execution turns.  The agent's own
    model is used for planning and recovery.

    Stateless. Derives phase from ``request.messages`` on every call.

    Best results when the fast model uses the same provider as the
    agent's primary model (e.g. both Anthropic).  Cross-provider routing
    works but provider-specific ``model_settings`` may need adjustment.

    Args:
        fast: Model identifier string (``"provider:model"`` format) or
            a pre-constructed ``BaseChatModel`` instance.

    Raises:
        ValueError: If *fast* is empty or cannot be resolved.

    Example::

        from langchain_router import RouterMiddleware

        middleware = [RouterMiddleware(fast="anthropic:claude-haiku-4-5-20251001")]
    """

    state_schema = AgentState

    def __init__(self, *, fast: str | BaseChatModel) -> None:
        """Create a new ``RouterMiddleware``.

        Raises:
            ValueError: If *fast* is empty or cannot be resolved.
        """
        if isinstance(fast, str):
            if not fast or not fast.strip():
                msg = "fast model identifier must not be empty"
                raise ValueError(msg)
            try:
                self._fast_model: BaseChatModel = init_chat_model(fast)
            except Exception as exc:
                msg = f"failed to resolve fast model {fast!r}: {exc}"
                raise ValueError(msg) from exc
        elif isinstance(fast, BaseChatModel):
            self._fast_model = fast
        else:
            msg = (
                f"fast must be a model string or BaseChatModel,"
                f" got {type(fast).__name__}"
            )
            raise TypeError(msg)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Route to the fast model during execution phases.

        Args:
            request: Model request to process.
            handler: Handler to call with the (possibly modified) request.

        Returns:
            Model response from *handler*.
        """
        phase = _detect_phase(request.messages)

        if phase == "execution":
            logger.debug("Phase: %s, routing to fast model", phase)
            return handler(request.override(model=self._fast_model))

        logger.debug("Phase: %s, using default model", phase)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Route to the fast model during execution phases.

        Args:
            request: Model request to process.
            handler: Async handler to call with the (possibly modified) request.

        Returns:
            Model response from *handler*.
        """
        phase = _detect_phase(request.messages)

        if phase == "execution":
            logger.debug("Phase: %s, routing to fast model", phase)
            return await handler(request.override(model=self._fast_model))

        logger.debug("Phase: %s, using default model", phase)
        return await handler(request)
