# TitleTrap Benchmark  
Probing Presentation Bias in LLM-Based Scientific Reviewing  

This repository accompanies the paper:  

**TitleTrap: Probing Presentation Bias in LLM-Based Scientific Reviewing**  
Shurui Du, University of Minnesota, Twin Cities (2025)  

---

## Overview  

TitleTrap is a controlled benchmark designed to test whether large language models (LLMs) such as GPT-4o and Claude exhibit presentation bias in scientific reviewing.  

Specifically, we study how the style of a paper’s title (branded, descriptive, or interrogative) affects LLM review judgments when the abstract remains fixed.  

Each benchmark item contains:  
- A research-style abstract (synthetic but plausible)  
- Three titles describing the same paper:  
  1. Branded / Colon-style  
  2. Plain Descriptive  
  3. Interrogative  

---

## Repository Structure  

```
titletrap-benchmark/
├── data/              # Generated benchmark data (JSON)
│   └── pairs/         # Abstract–title triplets for each field
├── prompts/           # Prompt templates for data generation & review
│   ├── datagen/
│   └── reviewer_prompt/
├── results/           # LLM review outputs (by model & round)
├── scripts/           # Core scripts for generation, review, and analysis
│   ├── gen_pairs.py       # Generate synthetic abstracts and titles
│   ├── demo.py            # Quick demonstration script
│   ├── review.py          # Run GPT/Claude reviews and collect outputs
│   └── plot_figures.ipynb # Analysis and visualization notebook
└── LICENSE            # MIT License
```

---

## Environment Setup  

1. Install dependencies  

```bash
pip install openai anthropic python-dotenv matplotlib seaborn pandas
```

2. Set API keys in `.env`:  

```
OPENAI_API_KEY=sk-xxxx
CLAUDE_API_KEY=sk-ant-xxxx
```

---

## Usage  

### 1. Generate Benchmark Data  

```bash
python scripts/gen_pairs.py --field cv
python scripts/review.py --field cv --round title-only --model gpt-4o
```

This creates files like:  

```
data/pairs/cv.json
data/pairs/nlp.json
```

Each file is an array of objects:  

```json
{
  "id": 1,
  "field": "CV",
  "title_a": "ImageFusion: Integrating Multi-Source Data for Enhanced Perception",
  "title_b": "ImageFusion for Enhanced Perception through Multi-Source Data Integration",
  "title_c": "Can ImageFusion Enhance Perception through Multi-Source Data Integration?",
  "abstract": "Introduces ImageFusion, a dual-stream framework..."
}
```

---

### 2. Run LLM Reviews  

Run the title-only or title+abstract evaluation:  

```bash
python scripts/review.py --field cv --round title-only --model gpt-4o
python scripts/review.py --field nlp --round title-abstract --model claude
```

Results are saved under:  

```
results/cv/title-only_gpt-4o.json
results/nlp/title-abstract_claude.json
```

---

### 3. Analyze Results  

Open the Jupyter notebook:  

```bash
jupyter notebook plot_figures.ipynb
```

This notebook reproduces the figures and tables from the paper, including:  
- Score distributions by title style  
- Model comparisons (GPT-4o vs. Claude)  
- Keyword polarity analysis from reviewer comments  

---

## Data Format Summary  

| Field | Description |
|-------|--------------|
| `id` | Unique numeric ID |
| `field` | Domain (CV / NLP / etc.) |
| `title_a` | Branded / Colon-style title |
| `title_b` | Plain Descriptive title |
| `title_c` | Interrogative title |
| `abstract` | Synthetic abstract |
| `scores` | (from results) Per-title LLM ratings |
| `choice` | Model’s preferred title |
| `reasons` | Textual justifications |

---

## License  

This project is released under the MIT License.  
See [LICENSE](LICENSE) for details.  