import os
import re
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    import anthropic
    claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
except ImportError:
    claude_client = None


def load_reviewer_prompt(round_name: str):
    """Load reviewer prompt for title-only or title-abstract."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "reviewer_prompt")
    prompt_file = os.path.join(base_path, f"{round_name}.txt")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def clean_and_parse_json(raw: str):
    """Clean model output and try to parse JSON."""
    cleaned = re.sub(r"^```json", "", raw.strip())
    cleaned = re.sub(r"^```", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("JSON parse failed, saving raw text instead.")
        return {"raw_output": raw.strip()}


def review_pair(pair, round_name="title-only", model="gpt-4o"):
    """Run review on a single pair with specified model and round."""
    prompt_template = load_reviewer_prompt(round_name)

    # Construct input text depending on round
    if round_name == "title-only":
        text = f"""
Title A: {pair['title_a']}
Title B: {pair['title_b']}
Title C: {pair['title_c']}
"""
    else:  # title-abstract
        text = f"""
Title A: {pair['title_a']}
Title B: {pair['title_b']}
Title C: {pair['title_c']}
Shared Abstract: {pair['abstract']}
"""

    if model.startswith("claude"):
        if claude_client is None:
            raise RuntimeError("Claude client not available. Install anthropic and set CLAUDE_API_KEY.")
        resp = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt_template + "\n\n" + text}]
        )
        raw_output = resp.content[0].text.strip()
    else:  # OpenAI GPT family
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_template + "\n\n" + text}],
            temperature=0.0,
        )
        raw_output = resp.choices[0].message.content.strip()

    return clean_and_parse_json(raw_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=True,
                        choices=["nlp", "cv", "bio", "climate", "social"],
                        help="Domain / field to review")
    parser.add_argument("--round", type=str, required=True,
                        choices=["title-only", "title-abstract"],
                        help="Review round type")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use: e.g. claude or gpt-4o / gpt-4o-mini")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of pairs to review (default: all)")
    args = parser.parse_args()

    infile = os.path.join(os.path.dirname(__file__), "..", "data", "pairs", f"{args.field}.json")
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", args.field)
    outfile = os.path.join(outdir, f"{args.round}_{args.model}.json")

    with open(infile, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if args.num:
        pairs = pairs[:args.num]

    results = []
    for pair in pairs:
        print(f"=== Reviewing Pair {pair['id']} ({args.round}, {args.model}) ===")
        res = review_pair(pair, round_name=args.round, model=args.model)
        res["id"] = pair["id"]
        results.append(res)
        print(res)

    os.makedirs(outdir, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    main()
