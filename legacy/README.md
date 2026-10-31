# v1.0 pipeline (original submission), kept for inspection

These scripts produced the tables, macros and figures of the original submission. They are superseded by the v2.0
pipeline in the repository root and are not used by it.

* `agg.py` contains the error that made the coverage column of Table 10 of the original submission wrong: `metrics()`
  inferred the active reserve rule from the file name (`'_online_'`, `'_std_'`, ...) and fell back to the online rule
  for the reserve-sweep files (`par_*_fixed1.2_*`), so coverage was computed from the shadow online bound (leg-log
  column 5) instead of the active fixed margin (column 15).
* `mktablesA.py` wrote the dial table from `pa/data/dial.csv`; that table had not been regenerated after the final dial
  evaluation, whose values were used in the text through `pa/data/macros.tex`.
* `uavenv_v1.py` and `evallib_v1.py` are the v1.0 simulator and evaluation library; `tests/test_equivalence.py` checks
  that the v2.0 simulator reproduces v1.0 dynamics exactly.
* `figs/*.m` are the Octave figure scripts of v1.0; `run_all_v1.sh` is the v1.0 pipeline.

## runs_v1/
`crestret_s0_online.pkl` is the CREST-R checkpoint (seed 0, trained with the online conformal reserve) used only in the ablations of the original submission. It is not used in v2.0, where CREST-R is trained with ten seeds under the fixed reserve (`runs/crestret_s*.pkl`).
