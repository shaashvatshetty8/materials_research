#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


SISSO_TEMPLATE = """!================================================================
! Auto-generated SISSO.in for metal/nonmetal replication tuning
!================================================================
ptype=2
ntask=1
desc_dim=2
nsample=(188,111)
restart=0

fstore=1
nsf=7
ops='{ops}'
fcomplexity=3
funit={funit}
fmax_min=1e-3
fmax_max=1e5
nf_sis={nf_sis}

method_so='L0'
nmodel=100
isconvex=(1,1)
bwidth=0.001
"""


@dataclass
class RunResult:
    name: str
    nf_sis: int
    funit: str
    ops: str
    returncode: int
    overlap: int | None
    no_count: int | None
    total: int | None
    accuracy: float | None
    run_dir: str


def parse_overlap(top_file: Path) -> int | None:
    if not top_file.exists():
        return None
    for line in top_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('rank'):
            continue
        parts = s.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def parse_accuracy(desc_file: Path) -> tuple[int | None, int | None, float | None]:
    if not desc_file.exists():
        return None, None, None
    total, no_count = 0, 0
    for line in desc_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('index'):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        total += 1
        if parts[1].upper() == 'NO':
            no_count += 1
    if total == 0:
        return None, None, None
    return no_count, total, (total - no_count) / total


def run_one(sisso_bin: Path, train_dat: Path, out_root: Path, np: int, nf_sis: int, funit: str, ops: str) -> RunResult:
    name = f"nf{nf_sis}_u{funit.replace('(', '').replace(')', '').replace(':', '-').replace(',', '_').replace(' ', '')}"
    run_dir = out_root / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(train_dat, run_dir / 'train.dat')
    (run_dir / 'SISSO.in').write_text(SISSO_TEMPLATE.format(nf_sis=nf_sis, funit=funit, ops=ops))

    log_path = run_dir / 'log'
    cmd = [str(sisso_bin)] if np <= 1 else ['mpirun', '-np', str(np), str(sisso_bin)]

    with log_path.open('w') as logf:
        proc = subprocess.run(cmd, cwd=run_dir, stdout=logf, stderr=subprocess.STDOUT)

    overlap = parse_overlap(run_dir / 'Models' / 'top0100_D002')
    no_count, total, acc = parse_accuracy(run_dir / 'Models' / 'data_top1' / 'desc_D002.dat')

    return RunResult(
        name=name,
        nf_sis=nf_sis,
        funit=funit,
        ops=ops,
        returncode=proc.returncode,
        overlap=overlap,
        no_count=no_count,
        total=total,
        accuracy=acc,
        run_dir=str(run_dir),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='Tune SISSO config for metal/nonmetal replication.')
    ap.add_argument('--sisso-bin', type=Path, required=True)
    ap.add_argument('--train-dat', type=Path, default=Path('/Users/shaashvatshetty/metal-nonmetal/paper_replication/sisso_official/train.dat'))
    ap.add_argument('--out-root', type=Path, default=Path('/Users/shaashvatshetty/metal-nonmetal/paper_replication/runs') / f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument('--np', type=int, default=8, help='MPI processes')
    ap.add_argument('--nf-sis', type=int, nargs='+', default=[1000, 2000, 5000, 10000])
    ap.add_argument('--funit', type=str, nargs='+', default=['(1:2)(3:7)', '(1:7)'])
    ap.add_argument('--ops', type=str, default='(+)(-)(*)(/)(exp)(log)(^-1)(^2)(^3)(sqrt)(|-|)')
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for u in args.funit:
        for n in args.nf_sis:
            print(f"\\n=== Running nf_sis={n}, funit={u} ===")
            r = run_one(args.sisso_bin, args.train_dat, args.out_root, args.np, n, u, args.ops)
            results.append(r)
            print(
                f"return={r.returncode} overlap={r.overlap} NO={r.no_count}/{r.total} "
                f"acc={None if r.accuracy is None else f'{r.accuracy:.4f}'}"
            )

    summary = {
        'sisso_bin': str(args.sisso_bin),
        'train_dat': str(args.train_dat),
        'np': args.np,
        'ops': args.ops,
        'results': [asdict(r) for r in results],
    }
    summary_path = args.out_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))

    ranked = sorted([r for r in results if r.accuracy is not None], key=lambda x: x.accuracy, reverse=True)
    print('\\n=== Ranked by accuracy ===')
    for r in ranked:
        print(f"{r.name:28s} acc={r.accuracy:.4f} NO={r.no_count}/{r.total} overlap={r.overlap} rc={r.returncode}")
    print(f"\\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
