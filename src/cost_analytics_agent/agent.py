from collections.abc import Callable

import anthropic
from anthropic import beta_tool

from cost_analytics_agent.db import DB_PATH, QueryRunner
from cost_analytics_agent.tools import Toolbox

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
MAX_ITERATIONS = 15
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_TEMPLATE = """You are a cloud cost analyst answering questions about an AWS bill stored \
in FOCUS format.

Data covers {start} to {end}. "Today" is the day after {end}; resolve relative dates \
("last month", "this quarter") against that. Costs are USD. Use effective_cost unless the user \
asks about invoiced amounts, in which case use billed_cost.

Tables available to run_sql (DuckDB dialect, SELECT only, max {max_rows} rows):
{schema}

Notes on the data:
- daily_cost.charge_category is 'Usage' or 'Purchase'. Purchase rows are commitment fees with \
billed_cost but zero effective_cost; their cost is amortized into the Usage rows they cover.
- pricing_category is 'Standard' (on-demand), 'Committed' (covered by a reservation), or 'None'.
- commitment_utilization has one row per commitment per day; utilization is \
used_hours / (used_hours + unused_hours).
- usage_type may carry a region prefix (e.g. USW2-BoxUsage:c6i.xlarge).

How to work:
- Every number you state must come from a tool result in this conversation. Never estimate.
- Use run_sql for lookups and aggregations. Aggregate in SQL instead of pulling raw rows.
- For "why did cost change" questions, use explain_change on the window in question. If the \
default baseline (the 28 days before) overlaps another unusual period, pass an explicit clean \
baseline.
- For "anything unusual" questions, use detect_anomalies, then explain_change on the windows \
that matter. The detector already expects the recurring weekly and day-of-month patterns \
(including the month-end batch bump), so flagged days are deviations from those. Days right \
after a multi-day spike can be flagged low because the trend lags; do not report those as \
separate incidents.
- For projections, use forecast_cost and report the 80% interval alongside the point estimate.
- Answer concisely: lead with the answer, then the supporting figures. State the date window \
you used. The billing data shows what changed, not why someone changed it; label any \
operational cause as a hypothesis.
"""


def build_system_prompt(toolbox: Toolbox) -> str:
    start, end = toolbox.date_range()
    schema = "\n".join(
        f"- {table}({', '.join(f'{col} {dtype}' for col, dtype in cols)})"
        for table, cols in toolbox.runner.schema().items()
    )
    return SYSTEM_TEMPLATE.format(
        start=start, end=end, schema=schema, max_rows=toolbox.runner.max_rows
    )


def build_tools(toolbox: Toolbox) -> list:
    @beta_tool
    def run_sql(sql: str) -> str:
        """Run one read-only DuckDB SELECT against the cost views and return the rows as JSON.

        Args:
            sql: A single SELECT statement over daily_cost, daily_total, monthly_cost, or
                commitment_utilization.
        """
        return toolbox.run_sql(sql)

    @beta_tool
    def detect_anomalies(
        start: str | None = None,
        end: str | None = None,
        account: str | None = None,
        service: str | None = None,
        region: str | None = None,
    ) -> str:
        """Find days whose cost deviates from the trend and weekly pattern.

        Returns the flagged days with actual cost, expected cost, and a robust z-score. Call
        this to locate unusual days before explaining them.

        Args:
            start: First date to report, YYYY-MM-DD. Defaults to the start of the data.
            end: Last date to report, YYYY-MM-DD. Defaults to the end of the data.
            account: Restrict to one account name, e.g. prod-core.
            service: Restrict to one service name, e.g. Amazon Simple Storage Service.
            region: Restrict to one region id, e.g. us-east-1.
        """
        return toolbox.detect_anomalies(start, end, account, service, region)

    @beta_tool
    def explain_change(
        current_start: str,
        current_end: str,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
    ) -> str:
        """Decompose the change in mean daily cost between a baseline window and a current window.

        Returns the top account/region/service/usage_type combinations driving the change and
        a breakdown per single dimension, each with its share of the total change.

        Args:
            current_start: First day of the window to explain, YYYY-MM-DD.
            current_end: Last day of the window to explain, YYYY-MM-DD.
            baseline_start: First day of the comparison window. Defaults to 28 days before
                current_start.
            baseline_end: Last day of the comparison window. Defaults to the day before
                current_start.
        """
        return toolbox.explain_change(current_start, current_end, baseline_start, baseline_end)

    @beta_tool
    def forecast_cost(
        horizon_days: int = 30,
        account: str | None = None,
        service: str | None = None,
        region: str | None = None,
    ) -> str:
        """Forecast daily cost past the end of the data, with an 80% interval.

        Picks between a seasonal-naive and an ETS model by backtest and returns the period
        total, weekly totals, and the backtest error of each model.

        Args:
            horizon_days: Number of days to forecast, 1 to 180.
            account: Restrict to one account name.
            service: Restrict to one service name.
            region: Restrict to one region id.
        """
        return toolbox.forecast_cost(horizon_days, account, service, region)

    return [run_sql, detect_anomalies, explain_change, forecast_cost]


class Refusal(RuntimeError):
    pass


class CostAgent:
    def __init__(
        self,
        runner: QueryRunner | None = None,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        effort: str = "medium",
    ):
        self.runner = runner or QueryRunner(DB_PATH)
        self.toolbox = Toolbox(self.runner)
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.tools = build_tools(self.toolbox)
        self.system = [
            {
                "type": "text",
                "text": build_system_prompt(self.toolbox),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self.messages: list[dict] = []

    def ask(self, question: str, on_tool_call: Callable[[str, dict], None] | None = None) -> str:
        self.messages.append({"role": "user", "content": question})
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=MAX_TOKENS,
            max_iterations=MAX_ITERATIONS,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        last = None
        for message in runner:
            last = message
            self.messages.append({"role": "assistant", "content": message.content})
            if on_tool_call:
                for block in message.content:
                    if block.type == "tool_use":
                        on_tool_call(block.name, block.input)
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)
        if last is None:
            raise RuntimeError("no response from the model")
        if last.stop_reason == "refusal":
            raise Refusal("the model declined to answer this request")
        text = "\n".join(b.text for b in last.content if b.type == "text").strip()
        if last.stop_reason == "max_tokens":
            text += "\n\n[response truncated at the token limit]"
        elif last.stop_reason == "tool_use":
            text += "\n\n[stopped after reaching the tool-call limit]"
        return text

    def reset(self) -> None:
        self.messages = []
