# Metal / Nonmetal Symbolic Regression

Symbolic regression experiments to discover a 2D analytical descriptor `(f1, f2)` that
linearly separates 299 binary inorganic compounds (188 metals, 111 nonmetals) using a
LinearSVC evaluated by leave-one-out cross-validation (LOOCV).

**Target:** beat the SISSO baseline from Ouyang et al. (2018, *Phys. Rev. Materials*)
which achieves **~0.987 LOOCV** on this dataset.

---

## Files

| File | Description |
|---|---|
| `github_style_mm_search.py` | Core search engine — data loading, LOOCV evaluation, LLMClient, LLM-SR and MCTS loops, prompt profiles |
| `prompt_evolution.py` | Prompt evolution runner: 5 prompts → top-2 → 5 new → repeat N generations |
| `llmsr.py` | Standalone custom LLM-SR script (early prototype) |
| `mcts.py` | Standalone custom MCTS script (early prototype) |
| `github_perov_runner.py` | Wrapper to run perovskite-repo search logic on this dataset |

---

## Dataset

Download from the Ouyang et al. supplementary materials and place at:
```
~/Downloads/metal-nonmetal_classification/
    binary_props.txt
    element_props.txt
```

---

## Usage

```bash
# set env vars
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://us.api.openai.com

# prompt evolution — LLM-SR mode, 4 generations, budget 20 LLM calls per profile
python prompt_evolution.py --mode llmsr --generations 4 --budget 20 --model gpt-4.1-nano

# prompt evolution — MCTS mode, 3 generations, budget 30
python prompt_evolution.py --mode mcts --generations 3 --budget 30 --model gpt-4.1-nano
```

Results are saved to `prompt_runs/<mode>_<timestamp>/`.

---

## Best result so far

**LOOCV = 0.9632** (vs. 0.987 paper baseline)

```python
f1 = packing * (X_B / X_A)   # geometry × electronegativity ratio
f2 = abs(IE_A - IE_B)         # ionization energy gap
```

Achieved by both LLM-SR (4-gen, budget 20) and MCTS (3-gen, budget 30) using the
`seed_1` prompt profile — a structured 6-section template covering problem context,
feature glossary, physical priors, formula patterns, safety rules, and a focus paragraph.

---

## Prompt evolution summary

| Run | Backend | Best LOOCV |
|---|---|---|
| Terse prompts, 3 gen | LLM-SR | 0.9331 |
| Descriptive prompts, 4 gen | LLM-SR | **0.9632** |
| Descriptive prompts, 3 gen | MCTS | **0.9632** |
| SISSO (paper) | — | 0.987 |

Key insight: adding a **feature glossary + physical priors + formula patterns** to the
prompt raised Gen-1 best from 0.877 → 0.963 (+8.5 points).
