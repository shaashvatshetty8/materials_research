"""
LLM-SR for Metal/Nonmetal Classification
=========================================
Uses GPT-4.1-nano to iteratively discover 2D symbolic descriptors that
outperform the SISSO paper's ~99% training / 97.6% LOOCV accuracy.

Run:
    OPENAI_API_KEY=sk-... python3 llmsr.py
"""

import os
import sys
import json
import time
import warnings
import textwrap
import requests
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR    = "/Users/shaashvatshetty/Downloads/metal-nonmetal_classification"
API_KEY     = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_file):
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("OPENAI_API_KEY="):
                    API_KEY = _line.split("=", 1)[1].strip()
                    break
MODEL       = "gpt-4.1-nano"
N_ITER      = 50          # LLM-SR iterations
TOP_K       = 5           # survivors per iteration
PROPOSALS   = 6           # expressions to request per LLM call
MAX_RETRIES = 3           # retries on API failure

# ─── Load data ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(f"{DATA_DIR}/binary_props.txt", sep=r"\s+", header=None)
    df.columns = ["Material","prototype","category","packing","dAB",
                  "CNA","CNB","xA","xB","dx","dy"]
    df["y"] = (df["category"] == "metal").astype(int)

    elem = pd.read_csv(f"{DATA_DIR}/element_props.txt", sep=r"\s+", header=None)
    elem.columns = ["Atom","IE","X","rcov","EAa","v"]
    lookup = elem.set_index("Atom")

    # Extract element symbols from Material formula
    from re import findall
    def elem_a(m):
        parts = findall(r"[A-Z][a-z]?", m)
        return parts[0] if parts else None
    def elem_b(m):
        parts = findall(r"[A-Z][a-z]?", m)
        return parts[1] if len(parts) > 1 else parts[0] if parts else None

    df["A"] = df["Material"].apply(elem_a)
    df["B"] = df["Material"].apply(elem_b)

    for prop in ["IE","X","rcov","EAa","v"]:
        df[f"{prop}_A"] = df["A"].map(lookup[prop])
        df[f"{prop}_B"] = df["B"].map(lookup[prop])

    return df

# ─── Evaluate a (f1, f2) expression pair ─────────────────────────────────────

EVAL_CONTEXT = {
    "sqrt": np.sqrt, "log": np.log, "ln": np.log,
    "abs":  np.abs,  "exp": np.exp, "np": np,
    "sin":  np.sin,  "cos": np.cos,
}

def compute_feature(df, expr):
    ctx = {**EVAL_CONTEXT}
    for col in df.select_dtypes(include=[np.number]).columns:
        ctx[col] = df[col].to_numpy(dtype=float)
    vals = eval(expr, {"__builtins__": None}, ctx)  # noqa: S307
    return np.array(vals, dtype=float)

def evaluate_pair(df, f1_expr, f2_expr):
    """Returns (train_acc, loocv_acc) or None on failure."""
    try:
        v1 = compute_feature(df, f1_expr)
        v2 = compute_feature(df, f2_expr)

        if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
            return None
        if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
            return None

        X  = np.column_stack([v1, v2])
        y  = df["y"].to_numpy()
        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        # Training accuracy
        clf = LinearSVC(C=1.0, max_iter=5000)
        clf.fit(Xs, y)
        train_acc = (clf.predict(Xs) == y).mean()

        # LOOCV accuracy
        n = len(y)
        loo_correct = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sc_loo = StandardScaler()
            Xtr = sc_loo.fit_transform(X[mask])
            Xte = sc_loo.transform(X[i:i+1])
            clf_loo = LinearSVC(C=1.0, max_iter=5000)
            clf_loo.fit(Xtr, y[mask])
            loo_correct += int(clf_loo.predict(Xte)[0] == y[i])
        loocv_acc = loo_correct / n

        return train_acc, loocv_acc
    except Exception:
        return None

def score(result):
    """Combined score: weight LOOCV heavily, break ties with training acc."""
    if result is None:
        return -1.0
    train_acc, loocv_acc = result
    return loocv_acc + 0.05 * train_acc

# ─── Baseline (SISSO) ────────────────────────────────────────────────────────

SISSO_F1 = "xA * packing * IE_B * sqrt(X_B) / X_A"
SISSO_F2 = "X_A**2 * abs(abs(1 - 2*xA) - xA**2 * X_B / X_A)"

def run_baseline(df):
    result = evaluate_pair(df, SISSO_F1, SISSO_F2)
    if result:
        print(f"\n{'='*60}")
        print("BASELINE (SISSO paper descriptor)")
        print(f"  f1 = {SISSO_F1}")
        print(f"  f2 = {SISSO_F2}")
        print(f"  Train accuracy : {result[0]:.4f}  ({result[0]*100:.1f}%)")
        print(f"  LOOCV accuracy : {result[1]:.4f}  ({result[1]*100:.1f}%)")
        print(f"{'='*60}\n")
    return result

# ─── OpenAI API call ─────────────────────────────────────────────────────────

ALL_FEATURES = [
    "packing","dAB","CNA","CNB","xA","xB",
    "IE_A","IE_B","X_A","X_B","rcov_A","rcov_B",
    "EAa_A","EAa_B","v_A","v_B",
]

FEATURE_DESCRIPTIONS = """
Available column names (all numerical):
  packing  - normalized packing fraction: sum(V_atom)/V_cell, V_atom = (4/3)pi*rcov^3
  dAB      - interatomic distance A-B in crystal (Angstrom)
  CNA      - coordination number of atom A
  CNB      - coordination number of atom B
  xA       - atomic fraction of element A  (xB = 1 - xA)
  xB       - atomic fraction of element B
  IE_A     - first ionization energy of element A (eV)
  IE_B     - first ionization energy of element B (eV)
  X_A      - Pauling electronegativity of element A
  X_B      - Pauling electronegativity of element B
  rcov_A   - covalent radius of element A (Angstrom)
  rcov_B   - covalent radius of element B (Angstrom)
  EAa_A    - electron affinity of element A (eV)
  EAa_B    - electron affinity of element B (eV)
  v_A      - number of valence electrons of element A
  v_B      - 8 minus valence electrons of element B (octet rule completion)

Allowed Python operators/functions: +, -, *, /, **, sqrt(), log(), abs(), exp(), sin(), cos()
"""

SYSTEM_PROMPT = """You are an expert in materials science and symbolic regression.
Your task: discover the best 2D symbolic descriptor (f1, f2) that linearly
separates METALS from NONMETALS for 299 binary compounds, evaluated by a
linear SVM with leave-one-out cross-validation (LOOCV).

OUTPUT FORMAT (STRICT)
- Return ONLY a JSON array of exactly {n} objects.
- Each object has EXACTLY two keys: "f1" and "f2", both Python expression strings.
- No prose. No markdown fences. No comments. No trailing text.

VALID VARIABLE NAMES (use ONLY these — DO NOT invent or rename)
  packing, dAB, CNA, CNB, xA, xB,
  IE_A, IE_B, X_A, X_B, rcov_A, rcov_B,
  EAa_A, EAa_B, v_A, v_B
  (typos like "packaging", "X_a", "ie_a" will be rejected)

ALLOWED OPERATORS / FUNCTIONS
  +  -  *  /  **  sqrt(.)  log(.)  abs(.)  exp(.)
  Numeric constants are allowed and ENCOURAGED for fine-tuning,
  e.g. 0.5, 1.7, 2.0, 0.25.

EXPRESSION SAFETY (avoid INVALID candidates)
- Never divide by a quantity that can be zero or negative without protection.
  If you need a difference in the denominator, write  abs(diff) + 1e-3 .
- Inside log(.) the argument must be > 0:  use  log(abs(x) + 1e-3).
- Inside sqrt(.) the argument must be >= 0:  use  sqrt(abs(x)).
- No conditionals, no loops, no list/dict literals.

STRATEGIC GUIDANCE (use materials physics)
- packing fraction ↑  =>  more interstitial electron charge => more metallic
- (X_B - X_A) large  =>  ionic / covalent => more likely nonmetal
- IE small            =>  easier to delocalize electrons => more metallic
- combine GEOMETRY (packing, dAB, rcov, CN) with CHEMISTRY (X, IE, EAa, v)
- non-linear forms (squares, sqrt, ratios, products of three terms) often
  separate classes that linear forms cannot.

DIVERSITY REQUIREMENT
- Every candidate in the same response must use a DIFFERENT structural form
  or a DIFFERENT subset of features. Do not return near-duplicates.

""" + FEATURE_DESCRIPTIONS


def build_cold_start_message():
    """First round: diverse fresh proposals, no seeding."""
    return textwrap.dedent(f"""
        ROUND 1 — COLD START.  No prior best descriptor.  Target: LOOCV > 0.987
        (the published reference accuracy for this dataset).

        Propose EXACTLY {PROPOSALS} (f1, f2) pairs that span DIFFERENT physical
        regimes.  Each pair must explore a distinct hypothesis.  Required mix:

          • Pair 1 — emphasis on PACKING FRACTION (geometry-driven metallicity)
          • Pair 2 — emphasis on ELECTRONEGATIVITY CONTRAST (X_B - X_A and ratios)
          • Pair 3 — emphasis on IONIZATION ENERGY (IE_A, IE_B, with composition xA)
          • Pair 4 — TRIPLE-PRODUCT form combining geometry + chemistry + composition
          • Pair 5 — RATIO form involving rcov and dAB (bond-length physics)
          • Pair 6 — your own novel hypothesis (be creative; can use exp/log)

        Use 4-8 operators per expression.  Use numeric constants where helpful.
        Apply the safety rules from the system prompt.  Return ONLY the JSON.
    """).strip()


def build_evolve_message(population, iteration):
    """Later rounds: explicitly ask for MUTATIONS and CROSSOVERS of top branches."""
    best_block = "\n".join(
        f"  branch {i+1}: f1={p['f1']!r}, f2={p['f2']!r}  "
        f"=> LOOCV={p['loocv']:.4f}, train={p['train']:.4f}"
        for i, p in enumerate(population)
    )
    best_loocv = max(p["loocv"] for p in population)
    gap_to_target = max(0.0, 0.987 - best_loocv)

    # Identify features already heavily used by current top branches
    used_features = set()
    for p in population:
        for col in ALL_FEATURES:
            if col in p["f1"] or col in p["f2"]:
                used_features.add(col)
    unused_features = [f for f in ALL_FEATURES if f not in used_features]
    unused_hint = (f"  Currently UNDERUSED features (try one of these "
                   f"for diversity): {unused_features}\n"
                   if unused_features else "")

    return textwrap.dedent(f"""
        Iteration {iteration}.  Current top-{len(population)} branches:
        {best_block}

        Best LOOCV so far: {best_loocv:.4f}.
        Gap to published reference (0.987): {gap_to_target:+.4f}.

        Generate EXACTLY {PROPOSALS} new (f1, f2) pairs.  Required composition:

          • 3 MUTATIONS — pick one parent branch above and make a SMALL,
            recognizable edit:
              - swap an operator (+ ↔ -, * ↔ /)
              - wrap one term with sqrt/log/abs
              - change an exponent (e.g. X_A**2 → X_A**1.5)
              - replace one feature with a closely related one
                (X_A ↔ X_B, IE_A ↔ IE_B, rcov_A ↔ rcov_B)
              - introduce a numeric constant (e.g. 0.5 * f, f + 0.25)

          • 2 CROSSOVERS — combine pieces from two DIFFERENT parent branches:
              - take f1 from branch i and f2 from branch j (i ≠ j)
              - OR splice a sub-expression from one branch into another

          • 1 EXPLORATION — must use at least one feature NOT present in any
            top branch above.{(' Try: ' + unused_hint.strip()) if unused_hint else ''}

        Goal: push LOOCV ABOVE the current best.  Apply all safety rules.
        Return ONLY the JSON array.
    """).strip()


def call_llm(user_message):
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(n=PROPOSALS)},
            {"role": "user",
             "content": user_message},
        ],
        "temperature": 0.9,
        "max_tokens": 1200,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip possible markdown fences
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            candidates = json.loads(content)
            return [(c["f1"], c["f2"]) for c in candidates if "f1" in c and "f2" in c]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [parse error attempt {attempt+1}]: {e}")
            time.sleep(1)
        except requests.HTTPError as e:
            print(f"  [HTTP error attempt {attempt+1}]: {e}")
            time.sleep(2)
    return []

# ─── LLM-SR evolutionary loop ────────────────────────────────────────────────

def llmsr_loop(df, baseline_result):
    # Start FROM SCRATCH — do NOT seed with SISSO baseline.
    population = []
    history = []

    print(f"Starting LLM-SR loop: {N_ITER} iterations, top-{TOP_K} survivors")
    print("Cold start: no seeding, no SISSO baseline in the population.\n")

    for it in range(1, N_ITER + 1):
        print(f"--- Iteration {it}/{N_ITER} ---")
        if not population:
            user_msg = build_cold_start_message()
            print("  [cold start: requesting diverse fresh proposals]")
        else:
            user_msg = build_evolve_message(population, it)
            print("  [evolving: requesting mutations + crossovers of top branches]")
        candidates = call_llm(user_msg)
        print(f"  Received {len(candidates)} candidates from LLM")

        new_results = []
        for f1, f2 in candidates:
            result = evaluate_pair(df, f1, f2)
            s = score(result)
            entry = {"f1": f1, "f2": f2,
                     "train": result[0] if result else 0.0,
                     "loocv": result[1] if result else 0.0,
                     "score": s}
            new_results.append(entry)
            status = f"LOOCV={result[1]:.4f} train={result[0]:.4f}" if result else "INVALID"
            print(f"    f1={f1[:50]!r}  =>  {status}")

        history.extend(new_results)

        # Select top-K by score
        population = sorted(
            [e for e in history if e["score"] > 0],
            key=lambda e: e["score"],
            reverse=True
        )[:TOP_K]

        if population:
            best = population[0]
            print(f"  Best so far: LOOCV={best['loocv']:.4f}  train={best['train']:.4f}")
            print(f"    f1 = {best['f1']}")
            print(f"    f2 = {best['f2']}\n")
        else:
            print("  No valid candidates yet, will retry next iteration.\n")

        time.sleep(0.5)  # avoid rate limits

    return population, history

# ─── Results summary ─────────────────────────────────────────────────────────

def print_results(df, population, history, baseline_result):
    print("\n" + "="*70)
    print("FINAL RESULTS — TOP 10 DISCOVERED DESCRIPTORS")
    print("="*70)

    top10 = sorted(
        [e for e in history if e["score"] > 0],
        key=lambda e: e["score"],
        reverse=True
    )[:10]

    b_train, b_loocv = baseline_result
    print(f"\n{'Rank':<5} {'LOOCV':>7} {'Train':>7}  f1 / f2")
    print("-"*70)
    for i, e in enumerate(top10):
        tag = " <-- SISSO baseline" if (
            e["f1"] == SISSO_F1 and e["f2"] == SISSO_F2) else ""
        print(f"  {i+1:<3}  {e['loocv']:.4f}  {e['train']:.4f}  {tag}")
        print(f"       f1 = {e['f1']}")
        print(f"       f2 = {e['f2']}")
        print()

    best = top10[0]
    print("="*70)
    print("BEST DISCOVERED DESCRIPTOR")
    print(f"  f1 = {best['f1']}")
    print(f"  f2 = {best['f2']}")
    print(f"  Train accuracy : {best['train']:.4f}  ({best['train']*100:.1f}%)")
    print(f"  LOOCV accuracy : {best['loocv']:.4f}  ({best['loocv']*100:.1f}%)")
    print()
    print("COMPARED TO SISSO BASELINE")
    print(f"  Train accuracy : {b_train:.4f}  ({b_train*100:.1f}%)")
    print(f"  LOOCV accuracy : {b_loocv:.4f}  ({b_loocv*100:.1f}%)")

    delta_loocv = best["loocv"] - b_loocv
    delta_train = best["train"] - b_train
    print()
    if delta_loocv > 0:
        print(f"  => LOOCV improved by {delta_loocv*100:+.2f}%  "
              f"(train: {delta_train*100:+.2f}%)")
    else:
        print(f"  => LOOCV did not improve ({delta_loocv*100:+.2f}%), "
              f"train: {delta_train*100:+.2f}%")

    # Misclassification analysis for best descriptor
    print("\n--- Misclassification Analysis (best descriptor) ---")
    v1 = compute_feature(df, best["f1"])
    v2 = compute_feature(df, best["f2"])
    X  = np.column_stack([v1, v2])
    y  = df["y"].to_numpy()
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(Xs, y)
    y_pred = clf.predict(Xs)
    wrong = df[y_pred != y][["Material", "prototype", "category"]].copy()
    wrong["predicted"] = np.where(y_pred[y_pred != y] == 1, "metal", "nonmetal")
    if len(wrong) == 0:
        print("  Perfect training classification! (0 misclassified)")
    else:
        print(f"  {len(wrong)} misclassified materials:")
        print(wrong.to_string(index=False))

    print("="*70)

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("ERROR: No API key found.\n"
              "  Option 1: export OPENAI_API_KEY=sk-...\n"
              "  Option 2: create ~/metal-nonmetal/key.txt containing your key")
        sys.exit(1)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} materials "
          f"({df['y'].sum()} metals, {(df['y']==0).sum()} nonmetals)")

    baseline_result = run_baseline(df)
    if baseline_result is None:
        print("ERROR: Baseline evaluation failed.")
        sys.exit(1)

    population, history = llmsr_loop(df, baseline_result)
    print_results(df, population, history, baseline_result)

if __name__ == "__main__":
    main()
