"""Terminal chat with the cost agent.

Usage: uv run cost-agent            (interactive)
       uv run cost-agent "question"  (one-shot)
"""

import json
import sys

import anthropic

from cost_analytics_agent.agent import CostAgent, Refusal
from cost_analytics_agent.db import DB_PATH, PARQUET, build_database


def show_tool_call(name: str, args: dict) -> None:
    print(f"  > {name} {json.dumps(args)}", file=sys.stderr)


def answer(agent: CostAgent, question: str) -> None:
    try:
        print(agent.ask(question, on_tool_call=show_tool_call))
    except Refusal as e:
        print(f"refused: {e}")
    except anthropic.AuthenticationError:
        raise SystemExit("authentication failed: set ANTHROPIC_API_KEY") from None
    except anthropic.RateLimitError:
        print("rate limited; try again shortly")
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        print("could not reach the API")


def main() -> None:
    if not DB_PATH.exists():
        if not PARQUET.exists():
            raise SystemExit("no data: run `uv run python -m cost_analytics_agent.generate` first")
        build_database()
    agent = CostAgent()
    if len(sys.argv) > 1:
        answer(agent, " ".join(sys.argv[1:]))
        return
    start, end = agent.toolbox.date_range()
    print(f"Cost data {start} to {end}. /reset clears the conversation, /quit exits.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            break
        if question == "/reset":
            agent.reset()
            continue
        answer(agent, question)


if __name__ == "__main__":
    main()
