import os
import re
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt(field):
    """Load datagen prompt for the given field."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "datagen")
    prompt_file = os.path.join(base_path, f"{field}_prompt.txt")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

def safe_load_json(content: str):
    """Clean and parse model output into strict JSON (no trailing commas)."""
    s = content.strip()

    # Strip markdown fences like ```json ... ```
    s = re.sub(r"^```[\w-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Keep only the outermost JSON array if extra text exists
    if "[" in s and "]" in s:
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]

    # Helper: remove trailing commas before } or ]
    def remove_trailing_commas(text: str) -> str:
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    s = remove_trailing_commas(s)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        print("JSON decode failed after cleaning, saving raw output instead.")
        return content

def generate_pairs(field, model="gpt-4o-mini"):
    """Generate 20 paper triplets (3 titles + abstract) for the given field."""
    prompt = load_prompt(field)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a strict JSON generator. "
                "Output ONLY a valid JSON array conforming to RFC 8259. "
                "Do not include comments, trailing commas, or any extra text."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    content = resp.choices[0].message.content.strip()
    return safe_load_json(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=True,
                        choices=["nlp", "cv", "bio", "climate", "social"],
                        help="Domain name")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    field = args.field
    outfile = args.outfile or os.path.join(
        os.path.dirname(__file__), "..", "data", "pairs", f"{field}.json"
    )

    print(f"Generating pairs for {field.upper()} ...")
    pairs = generate_pairs(field)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        if isinstance(pairs, str):
            f.write(pairs)  # raw text fallback
        else:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {field} pairs to {outfile}")

if __name__ == "__main__":
    main()
