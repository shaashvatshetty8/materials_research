"""
Run the official Perovskite-repo search implementations (LLM-SR / MCTS)
from this project without rewriting core search logic.

This script imports modules directly from:
  /Users/shaashvatshetty/ResearchFolder/materials_research

Usage examples:
  python3 github_perov_runner.py --mode llmsr
  python3 github_perov_runner.py --mode mcts
  python3 github_perov_runner.py --mode llmsr --budget 50
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PEROV_REPO = Path("/Users/shaashvatshetty/ResearchFolder/materials_research")
if str(PEROV_REPO) not in sys.path:
    sys.path.insert(0, str(PEROV_REPO))

# Imports below are from the Perovskite repo (official code path)
from evaluator import load_dataset  # type: ignore  # noqa: E402
from llm_client import LLMClient  # type: ignore  # noqa: E402
from llmsr import run_llmsr  # type: ignore  # noqa: E402
from mcts import run_mcts  # type: ignore  # noqa: E402
from state import SearchState  # type: ignore  # noqa: E402


@dataclass
class LLMCfg:
    model: str = "gpt-4.1-nano"
    temperature: float = 0.7
    max_tokens: int = 3000


@dataclass
class MCTSCfg:
    budget: int = 50
    initial_samples: int = 8
    ucb_constant: float = 1.41
    max_depth: int = 8
    max_children_per_node: int = 4


@dataclass
class EvalCfg:
    data_path: str = "perovskite-stability/TableS1.csv"
    decision_tree_max_depth: int = 3
    train_split_label: int = 1
    cv_folds: int = 5


class Cfg:
    def __init__(self) -> None:
        self.llm = LLMCfg()
        self.mcts = MCTSCfg()
        self.eval = EvalCfg()


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llmsr", "mcts"], required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--data-path", type=str, default="perovskite-stability/TableS1.csv")
    parser.add_argument("--run-root", type=str, default="/Users/shaashvatshetty/metal-nonmetal/github_runs")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        raise SystemExit("No OPENAI_API_KEY found (env or .env)")

    cfg = Cfg()
    cfg.llm.model = args.model
    cfg.mcts.budget = args.budget
    cfg.eval.data_path = args.data_path

    run_dir = Path(args.run_root) / args.mode / datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = run_dir / "plots"
    state_save_path = run_dir / "search_state.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = PEROV_REPO / cfg.eval.data_path
    df = load_dataset(str(data_path))
    state = SearchState()
    client = LLMClient(cfg, api_key)

    if args.mode == "llmsr":
        state = run_llmsr(client, state, df, plot_dir, cfg, state_save_path=state_save_path)
    else:
        state = run_mcts(client, state, df, plot_dir, cfg, state_save_path=state_save_path)

    top = state.top_k(10)
    print("\nTop formulas:")
    for i, node in enumerate(top, 1):
        print(f"{i:>2}. acc={node.accuracy:.4f} id={node.id} depth={node.depth}")
        if node.formula:
            print(f"    formula: {node.formula}")

    print("\nSearch complete")
    print(f"Budget used: {state.budget_used}/{cfg.mcts.budget}")
    print(f"LLM calls: {state.total_llm_calls} (debug: {state.debug_calls})")
    print(f"LLM usage: {client.usage_summary()}")
    print(f"State: {state_save_path}")


if __name__ == "__main__":
    main()
