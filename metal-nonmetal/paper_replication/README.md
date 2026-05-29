# SISSO paper replication (Ouyang et al. 2018)

Replicate 299-binary metal/nonmetal classification with [official SISSO](https://github.com/rouyang2017/SISSO).

## Quick start (server)

```bash
pip install pandas  # for build_train_dat.py
python paper_replication/build_train_dat.py   # edit DATA_DIR path if needed

cd paper_replication/sisso_official
# Copy or compile SISSO binary, then:
mpirun -np 8 /path/to/SISSO > log 2>&1
```

## Files

- `build_train_dat.py` — writes `sisso_official/train.dat` (7 features, 299 samples)
- `sisso_official/SISSO.in` — classification, 2D descriptor; **`nf_sis=1000`** for fast test runs

## Notes

- Paper reports **~99% training accuracy** (3 misclassified / 299).
- Compile SISSO on the server (Fortran + MPI). Mac gfortran build required local patches to upstream source.
- Do not commit `SIS_subspaces/`, compiled `SISSO` binary, or API keys.
