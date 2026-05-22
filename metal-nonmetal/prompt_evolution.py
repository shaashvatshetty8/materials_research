"""
Prompt evolution runner:
  - Start with 5 prompt profiles
  - Run each profile (MCTS or LLM-SR)
  - Keep top-2
  - Generate another 5 based on top-2
  - Repeat for N generations (default 3)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from github_style_mm_search import (
    PromptProfile,
    default_prompt_profiles,
    run_search_once,
)


def mutate_profile(base: PromptProfile, idx: int) -> PromptProfile:
    """Deterministic mutation fallback (no extra API dependencies)."""
    focus_bank = [
        (
            "COMPOSITION-RATIO MUTATION: anchor one axis on composition asymmetry "
            "(prefer xA*(1-xA), (1 - 2*xA), or xA**2 as multipliers) combined with a "
            "non-trivial electronegativity ratio such as X_B / X_A or log(1 + X_B/X_A). "
            "Anchor the other axis on a geometric quantity (packing, dAB, or rcov_A/rcov_B). "
            "Avoid descriptors whose two axes are scalar multiples of each other."
        ),
        (
            "COMPACT-NONLINEAR MUTATION: build short formulas (4-6 operators each) "
            "with exactly ONE ratio per axis and exactly ONE nonlinear transform "
            "(sqrt, log, exp, or **2). Example skeletons: f1 = sqrt(A) * (B/C), "
            "f2 = log(1 + D*E/F). Keep expressions interpretable; do NOT stack more "
            "than two sqrt/log/exp calls per axis."
        ),
        (
            "GEOMETRY+CHEMISTRY MUTATION: every axis MUST include at least one geometric "
            "feature (one of: packing, dAB, rcov_A, rcov_B, CNA, CNB) AND at least one "
            "chemistry feature (one of: X_A, X_B, IE_A, IE_B, EAa_A, EAa_B, v_A, v_B). "
            "Pure-geometry or pure-chemistry axes have plateaued near 0.95 LOOCV; the "
            "new descriptor must mix the two modalities multiplicatively or additively."
        ),
        (
            "ORTHOGONAL-AXES MUTATION: make f1 and f2 capture COMPLEMENTARY physics. "
            "If f1 emphasizes metallicity (packing, IE_A, v_A), then f2 should emphasize "
            "ionicity / charge transfer (|X_B - X_A|, EAa_B, IE_B/IE_A). The dominant "
            "variable in f1 must NOT be the dominant variable in f2. Aim for clear "
            "physical separation between the two axes."
        ),
        (
            "NUMERICAL-STABILITY MUTATION: aggressively guard against divide-by-zero "
            "and log/sqrt of nonpositive values. Wrap every denominator as "
            "(abs(expr) + 1e-3), every log as log(abs(expr) + 1e-3), and every sqrt as "
            "sqrt(abs(expr)). Prefer smooth, monotonic, bounded terms (e.g. exp(-|...|), "
            "log1p-style, or tanh-like 1 - 1/(1+|x|)) over polynomials of degree > 2."
        ),
        (
            "TRIPLE-PRODUCT MUTATION: experiment with the SISSO-style structure "
            "  axis = (composition_term) * (geometry_term) * (chemistry_ratio) "
            "for at least one of f1, f2. Example template: "
            "  f1 = xA * packing * sqrt(X_B) / X_A "
            "  f2 = (1 - xA) * IE_B / (IE_A + 1e-3) "
            "Keep the OTHER axis structurally different (e.g. an additive contrast)."
        ),
    ]
    focus = focus_bank[idx % len(focus_bank)]
    extra = (
        "\n\nMUTATION DIRECTIVE (this run only)\n"
        f"{focus}\n"
        "Also: do NOT return a descriptor structurally identical to the parent — "
        "change at least one operator family (ratio<->product, log<->sqrt, etc.) and "
        "at least one variable.\n"
    )
    return PromptProfile(
        name=f"{base.name}_mut{idx + 1}",
        system_prompt=base.system_prompt + " Strictly avoid duplicate candidates.",
        initial_template=base.initial_template + extra,
        improve_template=base.improve_template + extra,
    )


def generate_next_five(top2: list[PromptProfile], generation: int) -> list[PromptProfile]:
    # 2 direct elites + 3 mutations alternating parents
    p1, p2 = top2
    out = [
        PromptProfile(
            name=f"gen{generation}_elite1",
            system_prompt=p1.system_prompt,
            initial_template=p1.initial_template,
            improve_template=p1.improve_template,
        ),
        PromptProfile(
            name=f"gen{generation}_elite2",
            system_prompt=p2.system_prompt,
            initial_template=p2.initial_template,
            improve_template=p2.improve_template,
        ),
        mutate_profile(p1, 0),
        mutate_profile(p2, 1),
        mutate_profile(p1, 2),
    ]
    return out


def run_generation(
    profiles: list[PromptProfile],
    mode: str,
    budget: int,
    gen_dir: Path,
    model: str,
) -> list[dict]:
    results = []
    for i, profile in enumerate(profiles, start=1):
        run_dir = gen_dir / f"prompt_{i}_{profile.name}"
        outcome = run_search_once(mode=mode, profile=profile, budget=budget, run_dir=run_dir, model=model)
        results.append(
            {
                "profile_idx": i,
                "profile_name": profile.name,
                "profile": {
                    "system_prompt": profile.system_prompt,
                    "initial_template": profile.initial_template,
                    "improve_template": profile.improve_template,
                },
                "result": outcome,
            }
        )
        print(
            f"[{profile.name}] LOOCV={outcome['best_loocv']:.4f}, "
            f"Train={outcome['best_train']:.4f}, llm_calls={outcome.get('llm_calls', 0)}"
        )
    results.sort(key=lambda x: x["result"]["best_loocv"], reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llmsr", "mcts"], default="llmsr")
    parser.add_argument("--generations", type=int, default=3, help="Use 2 or 3 typically")
    parser.add_argument("--budget", type=int, default=20, help="Per prompt run budget")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--out", type=str, default="/Users/shaashvatshetty/metal-nonmetal/prompt_runs")
    args = parser.parse_args()

    root = Path(args.out) / f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)

    profiles = default_prompt_profiles()
    all_gen_reports = []

    for gen in range(1, args.generations + 1):
        print(f"\n=== Generation {gen}/{args.generations} ===")
        gen_dir = root / f"generation_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gen_results = run_generation(
            profiles=profiles,
            mode=args.mode,
            budget=args.budget,
            gen_dir=gen_dir,
            model=args.model,
        )
        all_gen_reports.append({"generation": gen, "results": gen_results})

        top2 = gen_results[:2]
        print(
            f"Top-2: {top2[0]['profile_name']} ({top2[0]['result']['best_loocv']:.4f}), "
            f"{top2[1]['profile_name']} ({top2[1]['result']['best_loocv']:.4f})"
        )

        # Prepare next 5 prompts from top-2 (except after final generation)
        if gen < args.generations:
            p1 = PromptProfile(name=top2[0]["profile_name"], **top2[0]["profile"])
            p2 = PromptProfile(name=top2[1]["profile_name"], **top2[1]["profile"])
            profiles = generate_next_five([p1, p2], generation=gen + 1)

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps({"mode": args.mode, "generations": all_gen_reports}, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()

