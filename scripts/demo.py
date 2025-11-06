import os
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_reviewer_prompt():
    """Load the reviewer prompt template from file."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
    prompt_file = os.path.join(base_path, "reviewer_prompt.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

def review_pair(pair, model="gpt-4o-mini"):
    """Send one pair of titles+abstract to the model for review."""
    prompt_template = load_reviewer_prompt()

    # Combine title and abstract for both A and B
    text_a = f"Title: {pair['title_a']}\nAbstract: {pair['abstract']}"
    text_b = f"Title: {pair['title_b']}\nAbstract: {pair['abstract']}"

    results = {}
    for label, text in [("A", text_a), ("B", text_b)]:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_template + "\n\n" + text}],
            temperature=0.0,
        )
        results[label] = resp.choices[0].message.content.strip()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=True,
                        choices=["nlp", "cv", "bio", "climate", "social"],
                        help="Select the data field")
    args = parser.parse_args()

    infile = os.path.join(os.path.dirname(__file__), "..", "data", "pairs", f"{args.field}.json")
    outfile = os.path.join(os.path.dirname(__file__), "..", "results", f"demo_{args.field}.json")

    with open(infile, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    results = {}
    for pair in pairs:
        print(f"=== Reviewing Pair {pair['id']} ===")
        res = review_pair(pair)
        results[pair['id']] = res
        print(res)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {outfile}")

if __name__ == "__main__":
    main()