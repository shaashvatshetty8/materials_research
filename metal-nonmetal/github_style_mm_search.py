"""
GitHub-style MCTS + LLM-SR for metal/nonmetal classification.

This file ports the architecture used in
`ResearchFolder/materials_research/{llmsr,mcts}/search.py` and adapts it to
the metal/nonmetal dataset and evaluator.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

log = logging.getLogger(__name__)

DATA_DIR = Path("/Users/shaashvatshetty/Downloads/metal-nonmetal_classification")
FEATURES = [
    "packing", "dAB", "CNA", "CNB", "xA", "xB",
    "IE_A", "IE_B", "X_A", "X_B", "rcov_A", "rcov_B",
    "EAa_A", "EAa_B", "v_A", "v_B",
]


@dataclass
class Proposal:
    f1: str
    f2: str
    explanation: str = ""
    formula: str = ""


@dataclass
class EvalResult:
    accuracy: float = 0.0  # LOOCV accuracy (primary)
    train_accuracy: float = 0.0
    metrics_summary: str = ""
    error: str = ""


@dataclass
class FormulaNode:
    id: str
    parent_id: str | None
    code: str  # JSON string with f1/f2
    description: str
    formula: str = ""
    accuracy: float = 0.0
    metrics: dict = field(default_factory=dict)
    plot_path: str = ""
    visit_count: int = 0
    total_reward: float = 0.0
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class SearchState:
    nodes: dict[str, FormulaNode] = field(default_factory=dict)
    root_children: list[str] = field(default_factory=list)
    budget_used: int = 0
    total_llm_calls: int = 0
    debug_calls: int = 0

    def add_node(self, node: FormulaNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            if node.id not in self.root_children:
                self.root_children.append(node.id)
        else:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)

    def recompute_ranks(self) -> dict[str, float]:
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1].accuracy, reverse=True)
        total = len(sorted_nodes)
        if total == 0:
            return {}
        return {nid: 1.0 - (rank / total) for rank, (nid, _node) in enumerate(sorted_nodes)}

    def top_k(self, k: int = 10) -> list[FormulaNode]:
        return sorted(self.nodes.values(), key=lambda n: n.accuracy, reverse=True)[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "root_children": self.root_children,
            "budget_used": self.budget_used,
            "total_llm_calls": self.total_llm_calls,
            "debug_calls": self.debug_calls,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SearchState":
        data = json.loads(path.read_text())
        state = cls()
        state.root_children = data["root_children"]
        state.budget_used = data["budget_used"]
        state.total_llm_calls = data["total_llm_calls"]
        state.debug_calls = data["debug_calls"]
        for nid, ndata in data["nodes"].items():
            state.nodes[nid] = FormulaNode(**ndata)
        return state


@dataclass
class PromptProfile:
    name: str
    system_prompt: str
    initial_template: str
    improve_template: str


_BASE_SYSTEM = (
    "You are a senior materials-science researcher and symbolic-regression expert "
    "helping discover a 2D analytical descriptor (f1, f2) that linearly separates "
    "299 binary inorganic compounds (188 metals + 111 nonmetals) using a linear SVM "
    "evaluated by leave-one-out cross-validation (LOOCV). "
    "The published SISSO reference for this dataset achieves ~0.987 LOOCV — your goal "
    "is to match or exceed it. "
    "Output STRICT JSON only — no markdown, no prose outside the JSON object."
)

_PROBLEM_BACKGROUND = (
    "PROBLEM CONTEXT\n"
    "- 299 binary materials AxBy where the prototype is one of 15 crystal structures.\n"
    "- A linear SVM is fit on the 2D space (f1, f2) and accuracy is measured by LOOCV.\n"
    "- A good (f1, f2) pair should be MONOTONIC with respect to metallicity in at least\n"
    "  one axis; the second axis ideally encodes orthogonal information (e.g. ionicity).\n"
)

_FEATURE_GLOSSARY = (
    "FEATURE GLOSSARY (use ONLY these variable names exactly, no others)\n"
    "  packing   : Σ V_atom / V_cell with V_atom = (4/3)π·rcov³  (dimensionless, ~0.5–2)\n"
    "  dAB       : interatomic A–B distance (Å, ~1.8–3.5)\n"
    "  CNA, CNB  : coordination numbers of A and B (integers, 2–12)\n"
    "  xA, xB    : atomic fractions, xA + xB = 1\n"
    "  IE_A, IE_B: first ionization energies (eV, ~3–25). Lower => electron-donor-like.\n"
    "  X_A, X_B  : Pauling electronegativities (~0.7–4.0). Large (X_B - X_A) => ionic.\n"
    "  rcov_A,   : covalent radii (Å, ~0.3–2.5)\n"
    "  rcov_B\n"
    "  EAa_A,    : electron affinities (eV, ~0–3.6)\n"
    "  EAa_B\n"
    "  v_A, v_B  : valence electrons (1–8)\n"
)

_PHYSICS_PRIORS = (
    "PHYSICAL PRIORS (proven useful in the SISSO paper and beyond)\n"
    "- Higher packing fraction => more interstitial charge density => more metallic.\n"
    "- Large electronegativity contrast (X_B - X_A) => ionic / nonmetal.\n"
    "- Low ionization energy IE_A => easy to delocalize electrons => more metallic.\n"
    "- Composition asymmetry through xA·(1-xA) or (1 - 2·xA) often separates AxB(1-x).\n"
    "- Geometry-only OR chemistry-only descriptors usually plateau ~0.95 LOOCV.\n"
    "  Strong descriptors COMBINE geometry (packing/dAB/rcov) with chemistry (X/IE).\n"
    "- The SISSO reference uses a 'triple product' like  xA · packing · IE_B · √X_B / X_A\n"
    "  — you do NOT need to copy this, but its structure (composition × geometry ×\n"
    "  chemistry-ratio) is a strong template family.\n"
)

_FORMULA_PATTERNS = (
    "USEFUL FORMULA PATTERNS (mix and match — do NOT copy verbatim)\n"
    "- Ratio:           A_quantity / B_quantity                e.g.  X_A / X_B,  IE_A / IE_B\n"
    "- Contrast:        |A_quantity - B_quantity|              e.g.  |X_B - X_A|\n"
    "- Triple product:  geom × chem × composition              e.g.  packing · sqrt(X_B) · xA\n"
    "- Soft nonlinearity:  log(1 + ...) or sqrt(abs(...))\n"
    "- Asymmetric envelope:  exp(-|X_A - X_B|)  or  xA · (1 - xA)\n"
    "- Charge-radius coupling:  IE_A · rcov_A  or  EAa_B / rcov_B\n"
)

_SAFETY_RULES = (
    "SAFETY / NUMERICAL RULES (violating these produces NaN/Inf and yields LOOCV = 0)\n"
    "- Never divide by a raw quantity that can be 0 or negative. Wrap:\n"
    "    /(expr)         =>  /(abs(expr) + 1e-3)\n"
    "- log argument must be > 0. Use:  log(abs(expr) + 1e-3).\n"
    "- sqrt argument must be ≥ 0. Use:  sqrt(abs(expr)).\n"
    "- No conditionals, list literals, lambdas, or imports — Python expressions only.\n"
    "- Each expression must be a SCALAR per row (no aggregates over the dataset).\n"
)

_OUTPUT_FORMAT = (
    "OUTPUT FORMAT (STRICT)\n"
    "Return ONLY a single JSON object with EXACTLY these three keys:\n"
    '  {{"f1": "<python expression>", "f2": "<python expression>", '
    '"explanation": "<1-2 sentences of physical reasoning>"}}\n'
    "Do NOT include keys like function/code/formula/markdown fences.\n"
)


def default_prompt_profiles() -> list[PromptProfile]:
    focuses = [
        (
            "packing + electronegativity contrast",
            "Combine the packing fraction (a geometry-driven metallicity proxy) with an "
            "electronegativity contrast term (e.g. |X_B - X_A| or X_B / X_A). One axis "
            "should be dominated by geometry, the other by chemistry."
        ),
        (
            "geometry + ionization-energy ratios",
            "Build one axis from a ratio or contrast of ionization energies (IE_A vs IE_B) "
            "modulated by a geometric quantity (dAB, rcov, packing). Aim for orthogonal "
            "geometry/chemistry axes."
        ),
        (
            "composition-aware nonlinearity",
            "Make composition (xA, xB) carry real signal, using forms like xA·(1-xA), "
            "(1 - 2·xA), or xA^2 · X_B / X_A. Combine with a geometry term so the model "
            "captures both stoichiometry and packing."
        ),
        (
            "valence and electron-affinity corrections",
            "Use v_A, v_B (valence electrons) and EAa_A, EAa_B (electron affinities) as "
            "first-class terms. Multiply or divide by electronegativity/IE so the descriptor "
            "is sensitive to charge-transfer character."
        ),
        (
            "balanced physically interpretable descriptor",
            "Produce a descriptor that a materials scientist could write on a napkin: "
            "ideally 4–7 operators per axis, mixing one geometry feature, one chemistry "
            "feature, and one composition feature. Prefer interpretability over complexity."
        ),
    ]

    templates = []
    for i, (focus_label, focus_block) in enumerate(focuses, start=1):
        initial = (
            f"{_PROBLEM_BACKGROUND}\n"
            f"{_FEATURE_GLOSSARY}\n"
            f"{_PHYSICS_PRIORS}\n"
            f"{_FORMULA_PATTERNS}\n"
            f"{_SAFETY_RULES}\n"
            f"FOCUS FOR THIS PROMPT: {focus_label}.\n"
            f"GUIDANCE: {focus_block}\n\n"
            "Brainstorm 3 candidates internally, pick the strongest, and return JUST it.\n"
            "Variables available: {vars}.\n\n"
            f"{_OUTPUT_FORMAT}"
        )
        improve = (
            f"{_PROBLEM_BACKGROUND}\n"
            f"{_FEATURE_GLOSSARY}\n"
            f"{_PHYSICS_PRIORS}\n"
            f"{_FORMULA_PATTERNS}\n"
            f"{_SAFETY_RULES}\n"
            "PARENT DESCRIPTOR TO IMPROVE\n"
            "  f1 = {parent_f1}\n"
            "  f2 = {parent_f2}\n"
            "  metrics: {metrics}\n\n"
            f"FOCUS FOR THIS PROMPT: {focus_label}.\n"
            f"GUIDANCE: {focus_block}\n\n"
            "Propose a NEW (f1, f2) that beats the parent on LOOCV. Constraints:\n"
            "- Must have a DIFFERENT functional structure than the parent (not just a "
            "  constant rescaling, sign flip, or trivial renaming).\n"
            "- Must include at least one geometry term AND one chemistry term.\n"
            "- Keep total operators per expression roughly between 4 and 8.\n"
            "Variables available: {vars}.\n\n"
            f"{_OUTPUT_FORMAT}"
        )
        templates.append(
            PromptProfile(
                name=f"seed_{i}",
                system_prompt=_BASE_SYSTEM,
                initial_template=initial,
                improve_template=improve,
            )
        )
    return templates


class LLMClient:
    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.calls = 0
        # Some accounts require the US regional hostname.
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com")

    def query_json(self, messages: list[dict]) -> dict:
        self.calls += 1
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=90,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning("API call failed: %s", e)
            return {}

        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]

        # Try strict parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract the largest plausible JSON object
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            blob = match.group(0)
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                # Light repair: strip stray backslashes that often break JSON
                repaired = re.sub(r"\\(?![\"\\/bfnrtu])", "", blob)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    log.warning("JSON parse failed; raw content head: %s", content[:120])
                    return {}

        log.warning("No JSON object found in response; head: %s", content[:120])
        return {}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "binary_props.txt", sep=r"\s+", header=None)
    df.columns = ["Material", "prototype", "category", "packing", "dAB", "CNA", "CNB", "xA", "xB", "dx", "dy"]
    df["y"] = (df["category"] == "metal").astype(int)

    elem = pd.read_csv(DATA_DIR / "element_props.txt", sep=r"\s+", header=None)
    elem.columns = ["Atom", "IE", "X", "rcov", "EAa", "v"]
    lookup = elem.set_index("Atom")

    def parse_formula(m: str) -> list[str]:
        return re.findall(r"[A-Z][a-z]?", m)

    df["A"] = df["Material"].map(lambda m: parse_formula(m)[0])
    df["B"] = df["Material"].map(lambda m: parse_formula(m)[1])

    for p in ["IE", "X", "rcov", "EAa", "v"]:
        df[f"{p}_A"] = df["A"].map(lookup[p])
        df[f"{p}_B"] = df["B"].map(lookup[p])
    return df


def _compute_feature(df: pd.DataFrame, expr: str) -> np.ndarray:
    ctx = {"np": np, "sqrt": np.sqrt, "log": np.log, "abs": np.abs, "exp": np.exp}
    for col in df.select_dtypes(include=[np.number]).columns:
        ctx[col] = df[col].to_numpy(dtype=float)
    return np.array(eval(expr, {"__builtins__": None}, ctx), dtype=float)  # noqa: S307


def evaluate_candidate(f1: str, f2: str, df: pd.DataFrame) -> EvalResult:
    try:
        v1 = _compute_feature(df, f1)
        v2 = _compute_feature(df, f2)
        if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
            return EvalResult(error="non-finite values")
        if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
            return EvalResult(error="near-constant feature")

        X = np.column_stack([v1, v2])
        y = df["y"].to_numpy()

        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        clf = LinearSVC(C=1.0, max_iter=5000)
        clf.fit(Xs, y)
        train = float((clf.predict(Xs) == y).mean())

        n = len(y)
        correct = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sc2 = StandardScaler()
            Xtr = sc2.fit_transform(X[mask])
            Xte = sc2.transform(X[i:i + 1])
            clf2 = LinearSVC(C=1.0, max_iter=5000)
            clf2.fit(Xtr, y[mask])
            correct += int(clf2.predict(Xte)[0] == y[i])
        loocv = correct / n
        return EvalResult(
            accuracy=loocv,
            train_accuracy=train,
            metrics_summary=f"LOOCV={loocv:.4f}, Train={train:.4f}",
        )
    except Exception as e:
        return EvalResult(error=str(e))


def _proposal_from_json(data: dict) -> Proposal:
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        data = {}
    return Proposal(
        f1=str(data.get("f1", "")).strip(),
        f2=str(data.get("f2", "")).strip(),
        explanation=str(data.get("explanation", "")).strip(),
        formula=str(data.get("formula", "")).strip(),
    )


def propose_initial(client: LLMClient, profile: PromptProfile) -> Proposal:
    prompt = profile.initial_template.format(vars=", ".join(FEATURES))
    messages = [{"role": "system", "content": profile.system_prompt}, {"role": "user", "content": prompt}]
    # Retry a few times to avoid empty/invalid JSON payloads.
    for _ in range(3):
        data = client.query_json(messages)
        p = _proposal_from_json(data)
        if p.f1 and p.f2:
            return p
    return Proposal(f1="", f2="", explanation="invalid proposal")


def propose_improvement(client: LLMClient, profile: PromptProfile, parent: FormulaNode) -> Proposal:
    prompt = profile.improve_template.format(
        parent_f1=json.loads(parent.code)["f1"],
        parent_f2=json.loads(parent.code)["f2"],
        metrics=parent.metrics.get("metrics_summary", ""),
        vars=", ".join(FEATURES),
    )
    messages = [{"role": "system", "content": profile.system_prompt}, {"role": "user", "content": prompt}]
    for _ in range(3):
        data = client.query_json(messages)
        p = _proposal_from_json(data)
        if p.f1 and p.f2:
            return p
    return Proposal(f1="", f2="", explanation="invalid proposal")


def _node_from_result(node_id: str, parent_id: str | None, proposal: Proposal, result: EvalResult, depth: int) -> FormulaNode:
    code = json.dumps({"f1": proposal.f1, "f2": proposal.f2})
    return FormulaNode(
        id=node_id,
        parent_id=parent_id,
        code=code,
        description=proposal.explanation,
        formula=proposal.formula or f"f1={proposal.f1}; f2={proposal.f2}",
        accuracy=result.accuracy,
        metrics={
            "train_accuracy": result.train_accuracy,
            "metrics_summary": result.metrics_summary,
        },
        visit_count=1,
        total_reward=result.accuracy,
        depth=depth,
    )


def _ucb1(node: FormulaNode, parent_visits: int, c: float) -> float:
    if node.visit_count == 0:
        return float("inf")
    exploitation = node.total_reward / node.visit_count
    exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / node.visit_count)
    return exploitation + exploration


def _select_node(state: SearchState, cfg) -> FormulaNode | None:
    if not state.root_children:
        return None
    total = sum(state.nodes[nid].visit_count for nid in state.root_children)
    current = state.nodes[max(state.root_children, key=lambda nid: _ucb1(state.nodes[nid], total, cfg.mcts.ucb_constant))]
    while current.depth < cfg.mcts.max_depth:
        if len(current.children_ids) < cfg.mcts.max_children_per_node:
            break
        current = state.nodes[
            max(current.children_ids, key=lambda nid: _ucb1(state.nodes[nid], current.visit_count, cfg.mcts.ucb_constant))
        ]
    return current


def _backpropagate(state: SearchState, node: FormulaNode) -> None:
    rewards = state.recompute_ranks()
    for nid, reward in rewards.items():
        n = state.nodes[nid]
        n.total_reward = reward * n.visit_count
    cur_id = node.parent_id
    while cur_id is not None:
        p = state.nodes[cur_id]
        p.visit_count += 1
        p.total_reward += rewards[node.id]
        cur_id = p.parent_id


def run_mcts(client: LLMClient, state: SearchState, df: pd.DataFrame, cfg, profile: PromptProfile, state_path: Path) -> SearchState:
    budget = cfg.mcts.budget
    initial = cfg.mcts.initial_samples
    while len(state.root_children) < initial and state.budget_used < budget:
        p = propose_initial(client, profile)
        state.total_llm_calls += 1
        r = evaluate_candidate(p.f1, p.f2, df)
        if not r.error:
            node = _node_from_result(uuid.uuid4().hex[:8], None, p, r, depth=0)
            state.add_node(node)
            _backpropagate(state, node)
        state.budget_used += 1
        state.save(state_path)

    while state.budget_used < budget:
        selected = _select_node(state, cfg)
        if selected is None:
            break
        p = propose_improvement(client, profile, selected)
        state.total_llm_calls += 1
        r = evaluate_candidate(p.f1, p.f2, df)
        if not r.error:
            node = _node_from_result(uuid.uuid4().hex[:8], selected.id, p, r, depth=selected.depth + 1)
            state.add_node(node)
            _backpropagate(state, node)
        state.budget_used += 1
        state.save(state_path)
    return state


def run_llmsr(client: LLMClient, state: SearchState, df: pd.DataFrame, cfg, profile: PromptProfile, state_path: Path) -> SearchState:
    budget = cfg.mcts.budget
    buf_size = 50
    sample_k = 3
    temp = 0.3

    buffer = [
        {"accuracy": n.accuracy, "node_id": n.id}
        for n in sorted(state.nodes.values(), key=lambda n: n.accuracy, reverse=True)
        if n.accuracy > 0
    ]
    iteration = state.budget_used

    while state.budget_used < budget:
        if len(buffer) < cfg.mcts.initial_samples:
            p = propose_initial(client, profile)
        else:
            scores = np.array([b["accuracy"] for b in buffer], dtype=float)
            w = np.exp((scores - scores.max()) / temp)
            w /= w.sum()
            k = min(sample_k, len(buffer))
            idx = np.random.choice(len(buffer), size=k, replace=False, p=w)
            parent = state.nodes[buffer[int(idx[0])]["node_id"]]
            p = propose_improvement(client, profile, parent)

        state.total_llm_calls += 1
        r = evaluate_candidate(p.f1, p.f2, df)
        if not r.error:
            node = _node_from_result(uuid.uuid4().hex[:8], None, p, r, depth=iteration)
            state.add_node(node)
            buffer.append({"accuracy": node.accuracy, "node_id": node.id})
            buffer.sort(key=lambda x: x["accuracy"], reverse=True)
            buffer = buffer[:buf_size]
        state.budget_used += 1
        iteration += 1
        state.save(state_path)
    return state


def get_api_key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "")
    if k:
        return k
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def make_cfg(mode: str, budget: int, model: str = "gpt-4.1-nano"):
    return SimpleNamespace(
        mode=mode,
        llm=SimpleNamespace(model=model, temperature=0.7, max_tokens=1800),
        mcts=SimpleNamespace(
            budget=budget,
            initial_samples=8,
            ucb_constant=1.41,
            max_depth=8,
            max_children_per_node=4,
        ),
    )


def run_search_once(mode: str, profile: PromptProfile, budget: int, run_dir: Path, model: str = "gpt-4.1-nano") -> dict:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY found in environment or .env")

    cfg = make_cfg(mode=mode, budget=budget, model=model)
    client = LLMClient(cfg.llm.model, cfg.llm.temperature, cfg.llm.max_tokens, api_key)
    df = load_data()
    state = SearchState()

    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "search_state.json"

    if mode == "mcts":
        state = run_mcts(client, state, df, cfg, profile, state_path)
    else:
        state = run_llmsr(client, state, df, cfg, profile, state_path)

    top = state.top_k(1)
    if not top:
        return {
            "best_loocv": 0.0,
            "best_train": 0.0,
            "best_f1": "",
            "best_f2": "",
            "state_path": str(state_path),
            "llm_calls": state.total_llm_calls,
            "budget_used": state.budget_used,
        }
    best = top[0]
    pair = json.loads(best.code)
    return {
        "best_loocv": best.accuracy,
        "best_train": float(best.metrics.get("train_accuracy", 0.0)),
        "best_f1": pair["f1"],
        "best_f2": pair["f2"],
        "state_path": str(state_path),
        "llm_calls": state.total_llm_calls,
        "budget_used": state.budget_used,
    }

