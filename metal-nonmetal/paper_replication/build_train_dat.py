"""Build train.dat for SISSO matching the paper's 299-binary metal/nonmetal
classification (Ouyang et al. 2018, Table I last row).

Primary features (in column order):
    IE_A, IE_B, chi_A, chi_B, xA, xB, packing
where packing = sum(V_atom) / V_cell  with V_atom = (4/3) pi rcov^3
(In the paper's table this column is written as Vcell/sum(Vatom), i.e. 1/packing.
 We follow the paper symbol directly: 'Vcell_over_Vatom' = 1/packing.)

train.dat layout (classification):
    materials  feat1  feat2  ...  featK
    <metals first, then nonmetals; rows grouped by class>
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

DATA_DIR = Path("/Users/shaashvatshetty/Downloads/metal-nonmetal_classification")
OUT = Path("/Users/shaashvatshetty/metal-nonmetal/paper_replication/sisso_official/train.dat")


def parse_formula(m: str) -> list[str]:
    return re.findall(r"[A-Z][a-z]?", m)


def main() -> None:
    binary = pd.read_csv(DATA_DIR / "binary_props.txt", sep=r"\s+", header=None)
    binary.columns = [
        "Material", "prototype", "category",
        "packing", "dAB", "CNA", "CNB", "xA", "xB", "dx", "dy",
    ]
    elem = pd.read_csv(DATA_DIR / "element_props.txt", sep=r"\s+", header=None)
    elem.columns = ["Atom", "IE", "X", "rcov", "EAa", "v"]
    lookup = elem.set_index("Atom")

    binary["A"] = binary["Material"].map(lambda m: parse_formula(m)[0])
    binary["B"] = binary["Material"].map(lambda m: parse_formula(m)[1])
    for p in ("IE", "X"):
        binary[f"{p}_A"] = binary["A"].map(lookup[p])
        binary[f"{p}_B"] = binary["B"].map(lookup[p])

    binary["Vcell_over_Vatom"] = 1.0 / binary["packing"]

    metals = binary[binary["category"] == "metal"].copy()
    nonmetals = binary[binary["category"] == "nonmetal"].copy()

    print(f"metals={len(metals)}  nonmetals={len(nonmetals)}  total={len(binary)}")

    feature_cols = ["IE_A", "IE_B", "X_A", "X_B", "xA", "xB", "Vcell_over_Vatom"]

    out_df = pd.concat([metals, nonmetals], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w") as fh:
        header = "materials  " + "  ".join(feature_cols) + "\n"
        fh.write(header)
        for _, row in out_df.iterrows():
            name = row["Material"]
            vals = "  ".join(f"{row[c]:.5f}" for c in feature_cols)
            fh.write(f"{name:<20s}  {vals}\n")

    print(f"Wrote {OUT}  rows={len(out_df)}  cols={1 + len(feature_cols)}")
    print(f"First class group (metals): {len(metals)}")
    print(f"Second class group (nonmetals): {len(nonmetals)}")


if __name__ == "__main__":
    sys.exit(main())
