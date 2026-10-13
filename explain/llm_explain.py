import json
import os

import pandas as pd
from dotenv import load_dotenv
from explain.strategy_summary import summarise_trace

# Load .env file
load_dotenv()


def load_baselines(path="data/processed/baselines_summary.csv"):
    df = pd.read_csv(path)
    # convert to a simple dict {strategy: total_time_s}
    return {row["strategy"]: float(row["total_time_s"]) for _, row in df.iterrows()}


def build_prompt(summary: dict) -> str:
    return f"""
You are an assistant that explains an F1 pit strategy chosen by a reinforcement learning agent.

Rules:
- Use ONLY the provided JSON. Do not invent numbers or tyre choices.
- When describing pit stops, you MUST use pit_plan_str exactly as written.
- If illegal_tyre_strategy is true, you MUST clearly say the strategy was illegal in a dry race because it failed the FIA two-compound rule.
- If tyre_rule_penalty_applied is greater than 0, mention that the final time includes this penalty.
- Do not claim the strategy is realistic or race-legal if illegal_tyre_strategy is true.

Write:
1) One-sentence headline using pit_plan_str
2) Whether the strategy was legal under the FIA dry-race tyre rule
3) Why each stop likely happened (degradation vs pit loss trade-off)
4) Comparison using exact times in baselines_total_time_s
5) Limitations note

Race summary (JSON):
{json.dumps(summary, indent=2)}
""".strip()


def main():
    # Build structured summary
    summary = summarise_trace("data/processed/ddqn_safety_car_eval_trace.csv")
    summary["baselines_total_time_s"] = load_baselines()
    prompt = build_prompt(summary)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set.\n"
            "Ensure you created a .env file with:\n"
            "OPENAI_API_KEY=your_key_here"
        )

    # Import here so error is clearer if missing
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Missing dependency. Install with: pip install openai")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    print("\n=== LLM Strategy Explanation ===\n")

    # Safer output extraction
    try:
        print(response.output_text)
    except AttributeError:
        # Fallback for structured output format
        for item in response.output:
            if item.type == "output_text":
                print(item.text)


if __name__ == "__main__":
    main()