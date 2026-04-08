"""Estimate cost savings from phase-based model routing.

Simulates a realistic agent session and shows per-session and annual
cost projections with and without RouterMiddleware.

Usage::

    python examples/benchmark.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from langchain_router import _detect_phase

# Model pricing (per million tokens, as of April 2026)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

# Average tokens per model call (assumed for cost projection)
_AVG_INPUT = 8_000
_AVG_OUTPUT = 1_500


def _build_session() -> list[list[AnyMessage]]:
    """Build model call snapshots for a realistic coding session."""
    snapshots: list[list[AnyMessage]] = []
    msgs: list[AnyMessage] = []
    cid = 0

    def _snap() -> None:
        snapshots.append(list(msgs))

    # User asks to fix a bug.
    msgs.append(HumanMessage(content="Fix the auth bug in the API"))
    _snap()

    # Agent reads 8 files.
    for name in [
        "auth.py",
        "models.py",
        "db.py",
        "routes.py",
        "config.py",
        "tests/test_auth.py",
        "tests/test_routes.py",
        "utils.py",
    ]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": f"/src/{name}"},
                        "id": f"tc{cid}",
                    }
                ],
            )
        )
        msgs.append(
            ToolMessage(content=f"# {name}\ndef main(): pass", tool_call_id=f"tc{cid}")
        )
        _snap()

    # Agent searches for 4 patterns.
    for pattern in ["verify_token", "session_expired", "401", "unauthorized"]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "grep", "args": {"pattern": pattern}, "id": f"tc{cid}"}
                ],
            )
        )
        msgs.append(
            ToolMessage(content=f"src/auth.py:42: {pattern}", tool_call_id=f"tc{cid}")
        )
        _snap()

    # Agent edits.
    cid += 1
    msgs.append(
        AIMessage(
            content="Found it.",
            tool_calls=[{"name": "edit_file", "args": {}, "id": f"tc{cid}"}],
        )
    )
    msgs.append(ToolMessage(content="File edited.", tool_call_id=f"tc{cid}"))
    _snap()

    # Test fails (recovery).
    cid += 1
    msgs.append(
        AIMessage(
            content="",
            tool_calls=[{"name": "execute", "args": {}, "id": f"tc{cid}"}],
        )
    )
    msgs.append(
        ToolMessage(
            content="Error: test_auth.py::test_login FAILED\nTraceback: AssertionError",
            tool_call_id=f"tc{cid}",
        )
    )
    _snap()

    # Agent fixes and re-runs.
    for content in ["File edited.", "All 12 tests passed."]:
        cid += 1
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file" if "edit" in content.lower() else "execute",
                        "args": {},
                        "id": f"tc{cid}",
                    }
                ],
            )
        )
        msgs.append(ToolMessage(content=content, tool_call_id=f"tc{cid}"))
        _snap()

    # User follow-up.
    msgs.append(HumanMessage(content="Can you also add a test for the edge case?"))
    _snap()

    return snapshots


def _cost_per_call(input_price: float, output_price: float) -> float:
    return (_AVG_INPUT * input_price + _AVG_OUTPUT * output_price) / 1_000_000


def main() -> None:
    """Run the benchmark."""
    snapshots = _build_session()

    counts: dict[str, int] = {"planning": 0, "execution": 0, "recovery": 0}
    for snap in snapshots:
        counts[_detect_phase(snap)] += 1

    total = len(snapshots)
    pct_routed = counts["execution"] / total * 100

    print("langchain-router benchmark")
    print("=" * 56)
    print(f"  Session: {total} model calls")
    p, e, r = counts["planning"], counts["execution"], counts["recovery"]
    print(f"  Planning: {p}  Execution: {e}  Recovery: {r}")
    print(f"  Routed to fast model: {pct_routed:.0f}%")
    print()

    for default_name, fast_name in [
        ("claude-sonnet-4-6", "claude-haiku-4-5"),
        ("gpt-5.4", "gpt-4.1-mini"),
    ]:
        d_in, d_out = _PRICING[default_name]
        f_in, f_out = _PRICING[fast_name]

        default_cost = _cost_per_call(d_in, d_out)
        fast_cost = _cost_per_call(f_in, f_out)

        without = total * default_cost
        with_ = (counts["planning"] + counts["recovery"]) * default_cost + counts[
            "execution"
        ] * fast_cost
        savings_pct = (1 - with_ / without) * 100

        # Annual projection: 10 devs, 20 sessions/day, 250 workdays
        sessions_year = 10 * 20 * 250
        annual_without = sessions_year * without
        annual_with = sessions_year * with_
        annual_saved = annual_without - annual_with

        print(f"  {default_name} -> {fast_name}")
        sess = f"${without:.4f} -> ${with_:.4f} ({savings_pct:.0f}% saved)"
        print(f"    Per session:  {sess}")
        print("    Annual (10 devs, 20 sessions/day):")
        print(f"      Without:  ${annual_without:>10,.0f}")
        print(f"      With:     ${annual_with:>10,.0f}")
        print(f"      Saved:    ${annual_saved:>10,.0f}/yr")
        print()


if __name__ == "__main__":
    main()
