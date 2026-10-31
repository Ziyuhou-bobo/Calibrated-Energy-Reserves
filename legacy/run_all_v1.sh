#!/bin/bash
# Full pipeline: training -> evaluation -> aggregation -> figures -> tables -> manuscript.
# Single CPU core is enough; total compute is roughly 6-7 hours (training dominates).
set -e
cd "$(dirname "$0")"
mkdir -p runs res logs figs/outA
# 1. training (25,000 environment steps, 16 parallel districts, 3 seeds; CREST-R single seed)
for s in 0 1 2; do
  for a in sac td3 ppo dsacn dsaccvar crest; do
    [ -f runs/${a}_s${s}.pkl ] || python3 train.py --algo $a --seed $s --steps 25000 --U 4 --out runs > logs/train_${a}_s${s}.log 2>&1
  done
done
[ -f runs/crestret_s0.pkl ] || python3 train.py --algo crestret --seed 0 --steps 25000 --U 4 --out runs > logs/train_crestret_s0.log 2>&1
# 2. evaluation on paired scenarios (resumable) and reserve sweep with the SAC policy
for s in 0 1 2; do
  python3 suite2.py $s all > logs/suite_s${s}.log 2>&1
  python3 par.py sac $s > logs/par_sac_s${s}.log 2>&1
done
# 3. qualitative data, aggregation, figure data and tables
python3 qualA.py
python3 agg2.py 0,1,2
python3 mkfigdata.py
python3 mktablesA.py
python3 checkmacros.py
# 4. figures (Octave, MATLAB-style output in EPS and PDF)
cd figs
for f in fig_schematic fig_map fig_power fig_learn fig_main fig_sev fig_dial fig_abl fig_alphaA fig_extra fig_cov fig_lam fig_pareto; do
  xvfb-run -a octave --no-gui --norc -q $f.m
done
cd ..
# 5. manuscript: inline all inputs, place floats next to their first citation, compile twice
python3 place_floats.py
mkdir -p build && cp /tmp/build_main.tex build/main.tex && cp figs/outA/*.pdf build/
cd build && pdflatex -interaction=nonstopmode main.tex > /dev/null && pdflatex -interaction=nonstopmode main.tex > /dev/null
echo "manuscript: build/main.pdf"
