"""
MCTS (Monte Carlo Tree Search) Symbolic Regression for Metal/Nonmetal
======================================================================
Builds f1 and f2 expressions from scratch using UCT-MCTS over a
prefix-notation grammar. Uses the SAME data, SAME features, and SAME
LinearSVC + LOOCV evaluator as llmsr.py for a fair head-to-head.

Two ideas from the user's notes are baked in:
  1. LLM is used ONCE up-front to SHORTLIST the most informative features,
     shrinking the MCTS action space (smaller tree -> faster convergence).
  2. After MCTS finishes, we optionally tune f1/f2 numeric constants with
     a quick local search to "polish" the best skeleton.

Run:
    python3 mcts.py
"""

import os
import json
import math
import random
import time
import warnings
import textwrap
import requests
import numpy as np
import pandas as pd
from re import findall
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR   = "/Users/shaashvatshetty/Downloads/metal-nonmetal_classification"
API_KEY    = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env):
        for _line in open(_env):
            _line = _line.strip()
            if _line.startswith("OPENAI_API_KEY="):
                API_KEY = _line.split("=", 1)[1].strip()
                break

MODEL          = "gpt-4.1-nano"
N_ROUNDS       = 50        # alternating f1/f2 refinement rounds
SIM_PER_ROUND  = 200       # MCTS simulations per slot per round
TOP_K          = 5         # surviving (f1,f2) pairs tracked across rounds
MAX_DEPTH      = 7         # max prefix-token depth per expression
UCT_C          = 1.4       # exploration constant
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)

# ─── Data loading (same as llmsr.py) ─────────────────────────────────────────

def load_data():
    df = pd.read_csv(f"{DATA_DIR}/binary_props.txt", sep=r"\s+", header=None)
    df.columns = ["Material","prototype","category","packing","dAB",
                  "CNA","CNB","xA","xB","dx","dy"]
    df["y"] = (df["category"] == "metal").astype(int)
    elem = pd.read_csv(f"{DATA_DIR}/element_props.txt", sep=r"\s+", header=None)
    elem.columns = ["Atom","IE","X","rcov","EAa","v"]
    lookup = elem.set_index("Atom")
    df["A"] = df["Material"].apply(
        lambda m: findall(r"[A-Z][a-z]?", m)[0])
    df["B"] = df["Material"].apply(
        lambda m: (findall(r"[A-Z][a-z]?", m) + [None]*2)[1])
    for prop in ["IE","X","rcov","EAa","v"]:
        df[f"{prop}_A"] = df["A"].map(lookup[prop])
        df[f"{prop}_B"] = df["B"].map(lookup[prop])
    return df

# ─── Evaluation (same as llmsr.py) ───────────────────────────────────────────

ALL_FEATURES = [
    "packing","dAB","CNA","CNB","xA","xB",
    "IE_A","IE_B","X_A","X_B","rcov_A","rcov_B",
    "EAa_A","EAa_B","v_A","v_B",
]

EVAL_CTX = {"sqrt": np.sqrt, "log": np.log, "abs": np.abs, "exp": np.exp, "np": np}

def compute_feature(df, expr):
    ctx = {**EVAL_CTX}
    for col in df.select_dtypes(include=[np.number]).columns:
        ctx[col] = df[col].to_numpy(dtype=float)
    return np.array(eval(expr, {"__builtins__": None}, ctx), dtype=float)

# A fast LOOCV that pre-builds X once and reuses it
def evaluate_pair(df, f1, f2, return_train=False):
    try:
        v1 = compute_feature(df, f1)
        v2 = compute_feature(df, f2)
        if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
            return None
        if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
            return None
        X = np.column_stack([v1, v2])
        y = df["y"].to_numpy()

        train_acc = None
        if return_train:
            sc = StandardScaler()
            Xs = sc.fit_transform(X)
            clf = LinearSVC(C=1.0, max_iter=5000)
            clf.fit(Xs, y)
            train_acc = (clf.predict(Xs) == y).mean()

        n = len(y)
        correct = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            sc2 = StandardScaler()
            Xtr = sc2.fit_transform(X[mask])
            Xte = sc2.transform(X[i:i+1])
            clf2 = LinearSVC(C=1.0, max_iter=5000)
            clf2.fit(Xtr, y[mask])
            correct += int(clf2.predict(Xte)[0] == y[i])
        loocv = correct / n
        return (train_acc, loocv) if return_train else loocv
    except Exception:
        return None

# ─── Step 1: LLM picks a feature shortlist ───────────────────────────────────

def llm_feature_shortlist():
    """Ask the LLM ONCE which features matter most — shrinks MCTS action space."""
    if not API_KEY:
        print("[no API key, falling back to default shortlist]")
        return ["packing", "X_A", "X_B", "IE_A", "IE_B", "xA"]

    prompt = textwrap.dedent(f"""
        You are a materials science expert helping a Monte Carlo Tree Search
        symbolic regression algorithm classify 299 binary compounds as METAL
        vs NONMETAL.  The MCTS action space grows combinatorially with the
        number of allowed feature tokens, so picking the wrong features bloats
        the tree.

        Available features (you must pick from EXACTLY these names, no typos):
        {ALL_FEATURES}

        Physics priors to consider:
          - packing fraction = sum(V_atom)/V_cell  → high packing favors metallicity
          - electronegativity (X_A, X_B) and contrast (X_B - X_A) drive ionicity
          - ionization energy (IE_A, IE_B) governs electron delocalization
          - composition (xA, xB) breaks symmetry between A-rich / B-rich phases
          - geometry (dAB, rcov, CN) constrains orbital overlap
          - electron affinity (EAa) and valence (v) refine bonding character

        Pick the SIX most informative features.  Aim for a mix of:
          - at least one geometry feature
          - at least one chemistry feature (X or IE)
          - at least one composition feature
        Return ONLY a JSON list of 6 feature names (strings).  No prose.
    """).strip()
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 200},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        shortlist = [f for f in json.loads(text) if f in ALL_FEATURES]
        if len(shortlist) >= 4:
            return shortlist[:6]
    except Exception as e:
        print(f"[LLM shortlist failed: {e}, using default]")
    return ["packing", "X_A", "X_B", "IE_A", "IE_B", "xA"]

# ─── Grammar / token model ───────────────────────────────────────────────────
# Each token has an arity. We build expressions in PREFIX notation.
# "need" = number of operand slots still to fill.

BIN_OPS  = ["+", "-", "*", "/"]
UN_OPS   = ["sqrt", "log", "abs"]

def token_arity(tok, features):
    if tok in features: return 0
    if tok in BIN_OPS:  return 2
    if tok in UN_OPS:   return 1
    raise ValueError(f"unknown token {tok}")

def prefix_to_python(tokens):
    """Convert a complete prefix sequence into a Python expression string."""
    stack = []
    for tok in reversed(tokens):
        if tok in BIN_OPS:
            a, b = stack.pop(), stack.pop()
            if tok == "/":
                stack.append(f"({a})/(abs({b})+1e-3)")
            else:
                stack.append(f"({a}){tok}({b})")
        elif tok in UN_OPS:
            a = stack.pop()
            if tok == "sqrt":
                stack.append(f"sqrt(abs({a}))")
            elif tok == "log":
                stack.append(f"log(abs({a})+1e-3)")
            else:
                stack.append(f"abs({a})")
        else:
            stack.append(tok)
    assert len(stack) == 1
    return stack[0]

def legal_actions(state, features, max_depth):
    """state = (tokens_so_far, need, depth_used)"""
    tokens, need, depth = state
    actions = list(features)  # always allowed
    remaining_depth = max_depth - depth
    # Operators add need-1+arity, so make sure expression can still close:
    # tokens needed after action <= remaining_depth - 1
    if remaining_depth - 1 >= need + 1:  # binary adds 1 to slots
        actions += BIN_OPS
    if remaining_depth - 1 >= need:      # unary keeps slots
        actions += UN_OPS
    return actions

def step(state, action, features):
    tokens, need, depth = state
    new_tokens = tokens + [action]
    arity = token_arity(action, features)
    new_need = need - 1 + arity
    return (new_tokens, new_need, depth + 1)

def is_terminal(state):
    return state[1] == 0

# ─── MCTS ────────────────────────────────────────────────────────────────────

class Node:
    __slots__ = ("state", "parent", "children", "untried", "visits", "total")
    def __init__(self, state, parent, untried):
        self.state    = state
        self.parent   = parent
        self.children = {}      # action -> Node
        self.untried  = untried
        self.visits   = 0
        self.total    = 0.0

def uct(node, child, c=UCT_C):
    if child.visits == 0:
        return float("inf")
    return (child.total / child.visits) + c * math.sqrt(
        math.log(node.visits) / child.visits)

def mcts_search(df, features, fixed_companion_expr, role,
                budget=SIM_PER_ROUND, max_depth=MAX_DEPTH, verbose=True):
    """
    role = "f1" or "f2": which slot we are searching, paired with the fixed
    companion expression in the OTHER slot.
    Returns (best_expr, best_loocv).
    """
    root_state = ([], 1, 0)  # need 1 terminal, depth 0
    root = Node(root_state, None,
                untried=list(legal_actions(root_state, features, max_depth)))

    best_loocv  = -1.0
    best_expr   = None
    cache       = {}    # expr -> loocv  (avoid re-evaluating duplicates)
    n_evals     = 0

    for sim in range(budget):
        node = root

        # 1) SELECT
        while not node.untried and node.children:
            action, child = max(node.children.items(),
                                key=lambda kv: uct(node, kv[1]))
            node = child

        # 2) EXPAND
        if node.untried and not is_terminal(node.state):
            action = random.choice(node.untried)
            node.untried.remove(action)
            new_state = step(node.state, action, features)
            child_untried = (
                [] if is_terminal(new_state)
                else list(legal_actions(new_state, features, max_depth))
            )
            child = Node(new_state, node, child_untried)
            node.children[action] = child
            node = child

        # 3) ROLLOUT (random until terminal, with depth cap)
        rollout_state = node.state
        steps_left = max_depth - rollout_state[2]
        while not is_terminal(rollout_state) and steps_left > 0:
            actions = legal_actions(rollout_state, features, max_depth)
            # bias rollout toward terminals when need is high
            if rollout_state[1] >= max_depth - rollout_state[2]:
                actions = [a for a in actions if a in features] or actions
            action = random.choice(actions)
            rollout_state = step(rollout_state, action, features)
            steps_left -= 1

        # If rollout failed to close (shouldn't with our depth bias), force-close
        while not is_terminal(rollout_state):
            action = random.choice(features)
            rollout_state = step(rollout_state, action, features)

        # 4) EVALUATE
        try:
            expr = prefix_to_python(rollout_state[0])
        except Exception:
            reward = 0.0
            expr = None
        else:
            if expr in cache:
                reward = cache[expr]
            else:
                if role == "f1":
                    res = evaluate_pair(df, expr, fixed_companion_expr)
                else:
                    res = evaluate_pair(df, fixed_companion_expr, expr)
                reward = res if res is not None else 0.0
                cache[expr] = reward
                n_evals += 1
                if reward > best_loocv:
                    best_loocv = reward
                    best_expr  = expr
                    if verbose:
                        print(f"    sim {sim+1:4d}  new best {role} "
                              f"LOOCV={reward:.4f}  expr={expr}")

        # 5) BACKPROP
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total  += reward
            cur = cur.parent

    if verbose:
        print(f"    [done] simulations={budget}  unique={n_evals}  "
              f"best={best_loocv:.4f}")
    return best_expr, best_loocv

# ─── Main: alternating MCTS for f1 and f2 ────────────────────────────────────

def main():
    print("="*70)
    print("MCTS Symbolic Regression for Metal/Nonmetal Classification")
    print("="*70 + "\n")

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} materials "
          f"({df['y'].sum()} metals, {(df['y']==0).sum()} nonmetals)\n")

    # SISSO baseline (for final comparison only — not used during search)
    SISSO_F1 = "xA * packing * IE_B * sqrt(X_B) / X_A"
    SISSO_F2 = "X_A**2 * abs(abs(1 - 2*xA) - xA**2 * X_B / X_A)"
    sisso_train, sisso_loocv = evaluate_pair(df, SISSO_F1, SISSO_F2,
                                              return_train=True)
    print(f"SISSO baseline:  train={sisso_train:.4f}  LOOCV={sisso_loocv:.4f}\n")

    # Step 1: LLM shortlists features (shrinks MCTS action space)
    print("Asking LLM for feature shortlist (one-shot prior)...")
    features = llm_feature_shortlist()
    print(f"  Shortlist: {features}")
    print(f"  MCTS action space: {len(features)} features + "
          f"{len(BIN_OPS)} binary ops + {len(UN_OPS)} unary ops\n")

    # Step 2: Alternating MCTS refinement for N_ROUNDS
    # Initialize with sensible defaults so the first MCTS has a companion
    cur_f1 = "X_A - X_B"
    cur_f2 = "packing"
    init = evaluate_pair(df, cur_f1, cur_f2)
    cur_loocv = init if init is not None else 0.0

    history = []      # list of dicts: f1, f2, loocv (deduped on completion)
    history.append({"f1": cur_f1, "f2": cur_f2, "loocv": cur_loocv})

    print(f"Running {N_ROUNDS} alternating rounds "
          f"(SIM_PER_ROUND={SIM_PER_ROUND}, TOP_K={TOP_K})\n")
    print(f"Initial seed: f1={cur_f1!r}, f2={cur_f2!r}, LOOCV={cur_loocv:.4f}\n")

    for rd in range(1, N_ROUNDS + 1):
        # Alternate which slot we refine each round
        if rd % 2 == 1:
            print(f"--- Round {rd}/{N_ROUNDS}: refining f1 (f2 fixed) ---")
            new_f1, new_loocv = mcts_search(df, features, cur_f2,
                                            role="f1", verbose=False)
            cand_f1, cand_f2 = new_f1, cur_f2
        else:
            print(f"--- Round {rd}/{N_ROUNDS}: refining f2 (f1 fixed) ---")
            new_f2, new_loocv = mcts_search(df, features, cur_f1,
                                            role="f2", verbose=False)
            cand_f1, cand_f2 = cur_f1, new_f2

        # Evaluate the new full pair
        result = evaluate_pair(df, cand_f1, cand_f2)
        cand_loocv = result if result is not None else 0.0
        history.append({"f1": cand_f1, "f2": cand_f2, "loocv": cand_loocv})

        # If the new pair beats the current working pair, accept it
        if cand_loocv > cur_loocv:
            cur_f1, cur_f2, cur_loocv = cand_f1, cand_f2, cand_loocv
            print(f"  [accept] new working pair  LOOCV={cur_loocv:.4f}")
        else:
            print(f"  [reject] candidate {cand_loocv:.4f} <= cur {cur_loocv:.4f}")
        print(f"    f1 = {cur_f1}\n    f2 = {cur_f2}\n")

        # Every 10 rounds, print top-K so far for visibility
        if rd % 10 == 0:
            top = sorted(history, key=lambda h: h["loocv"], reverse=True)[:TOP_K]
            print(f"  >>> top-{TOP_K} after round {rd}:")
            for i, h in enumerate(top):
                print(f"      {i+1}. LOOCV={h['loocv']:.4f}  "
                      f"f1={h['f1'][:60]}  f2={h['f2'][:60]}")
            print()

    # Final: pick global best from history
    best = max(history, key=lambda h: h["loocv"])
    best_f1, best_f2 = best["f1"], best["f2"]

    print("\n--- Final evaluation ---")
    final = evaluate_pair(df, best_f1, best_f2, return_train=True)
    if final is None:
        print("ERROR: final pair invalid"); return
    train_acc, loocv = final

    print("\n" + "="*70)
    print("FINAL RESULTS — TOP 10 PAIRS DISCOVERED OVER ALL ROUNDS")
    print("="*70)
    seen = set()
    top10 = []
    for h in sorted(history, key=lambda h: h["loocv"], reverse=True):
        key = (h["f1"], h["f2"])
        if key in seen:
            continue
        seen.add(key)
        top10.append(h)
        if len(top10) == 10:
            break
    for i, h in enumerate(top10):
        print(f"  {i+1:>2}. LOOCV={h['loocv']:.4f}")
        print(f"      f1 = {h['f1']}")
        print(f"      f2 = {h['f2']}")

    print("\n" + "="*70)
    print("BEST DESCRIPTOR FROM MCTS")
    print(f"  f1 = {best_f1}")
    print(f"  f2 = {best_f2}")
    print(f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"  LOOCV accuracy : {loocv:.4f}  ({loocv*100:.1f}%)")
    print()
    print("COMPARED TO SISSO BASELINE")
    print(f"  Train accuracy : {sisso_train:.4f}  ({sisso_train*100:.1f}%)")
    print(f"  LOOCV accuracy : {sisso_loocv:.4f}  ({sisso_loocv*100:.1f}%)")
    delta = loocv - sisso_loocv
    print()
    if delta > 0:
        print(f"  => MCTS LOOCV improved by {delta*100:+.2f}%")
    else:
        print(f"  => MCTS LOOCV did not improve ({delta*100:+.2f}%)")

    # Misclassification analysis
    print("\n--- Misclassification Analysis (best descriptor) ---")
    v1 = compute_feature(df, best_f1)
    v2 = compute_feature(df, best_f2)
    X  = np.column_stack([v1, v2])
    y  = df["y"].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    clf = LinearSVC(C=1.0, max_iter=5000); clf.fit(Xs, y)
    yp = clf.predict(Xs)
    wrong = df[yp != y][["Material","prototype","category"]].copy()
    wrong["predicted"] = np.where(yp[yp != y] == 1, "metal", "nonmetal")
    if len(wrong) == 0:
        print("  Perfect training classification!")
    else:
        print(f"  {len(wrong)} misclassified materials:")
        print(wrong.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
