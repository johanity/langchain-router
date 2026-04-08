# Verified Results

Routing tested with real API calls across 9 model pairs, 4 providers, 81 total calls. Every call routed correctly.

## Model Pairs Tested

| Primary | Fast | Planning | Execution | Recovery | Result |
|---------|------|----------|-----------|----------|--------|
| Claude Opus 4.6 | Claude Haiku 4.5 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| Claude Opus 4.6 | GLM-5 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| Claude Opus 4.6 | MiniMax M2.7 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| Claude Sonnet 4.6 | GLM-5 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| Claude Sonnet 4.6 | MiniMax M2.7 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| GPT-5.4 | GPT-4.1-mini | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| GPT-5.4 | GLM-5 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| GPT-5.4 | MiniMax M2.7 | ✓ primary | ✓ fast | ✓ primary | 9/9 |
| GLM-5 | MiniMax M2.7 | ✓ primary | ✓ fast | ✓ primary | 9/9 |

## Test Scenario

Each model pair ran a 9-call agent session simulating a developer investigating [langchain-ai/langchain#36491](https://github.com/langchain-ai/langchain/issues/36491) (`BaseTool.arun` pydantic v1 validation) with a follow-up on [#36492](https://github.com/langchain-ai/langchain/issues/36492) (`filter_messages` orphaned function_call blocks).

| Call | Phase | Messages | Expected |
|------|-------|----------|----------|
| plan | planning | User asks to investigate the `BaseTool.arun` bug | primary |
| exec1 | execution | Agent reads `langchain_core/tools/base.py` | fast |
| exec2 | execution | Agent reads `langchain_core/tools/structured.py` | fast |
| exec3 | execution | Agent reads `langchain_core/runnables/base.py` | fast |
| exec4 | execution | Agent reads `langchain/agents/middleware/types.py` | fast |
| exec5 | execution | Agent reads `tests/unit_tests/test_tools.py` | fast |
| recov | recovery | Test run fails with `ValidationError` traceback | primary |
| exec6 | execution | Agent edits fix, tool confirms success | fast |
| plan2 | planning | User asks about `filter_messages` issue | primary |

## Routing Patterns Verified

- Same provider: Anthropic → Anthropic, OpenAI → OpenAI
- Cross provider: Anthropic → Z.ai, Anthropic → MiniMax, OpenAI → Z.ai, OpenAI → MiniMax
- Frontier → open: Opus → GLM-5, Sonnet → MiniMax, GPT-5.4 → GLM-5
- Open → open: GLM-5 → MiniMax

## Models Used

Models tested:

- **Anthropic**: Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5
- **OpenAI**: GPT-5.4, GPT-4.1-mini
- **Z.ai**: GLM-5
- **MiniMax**: MiniMax M2.7

## Result

81/81 correct. 100% routing accuracy across all providers and model combinations.

Full per-call details: [full_verification.md](full_verification.md)
