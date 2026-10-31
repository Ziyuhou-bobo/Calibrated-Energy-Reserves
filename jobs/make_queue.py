"""Write jobs/queue.txt: every training and evaluation job of the revised study, in the order used.
Lines are 'OUTPUT | LOG | REQUIREMENTS | COMMAND'; jobs/runner3.py executes them, one core per job.
usage: python3 jobs/make_queue.py && python3 jobs/runner3.py jobs/queue.txt .
"""
import os

PY = 'python3'
MAINALG = ['sac', 'td3', 'dsacn', 'dsaccvar', 'crest', 'crestret']
PARTS = ['main', 'reserve', 'sweep', 'dispatch', 'dial', 'abl', 'sev', 'sub', 'cont']


def train_line(algo, s, arm):
    out = f'runs/{algo}_s{s}.pkl' if arm == 'fixed' else f'runs_online/{algo}_s{s}.pkl'
    od = 'runs' if arm == 'fixed' else 'runs_online'
    return (f'{out} | logs/train_{arm}_{algo}_s{s}.log | - | {PY} train.py --algo {algo} --seed {s} --steps 25000 --U 4 '
            f'--reserve {arm} --out {od}')


def eval_lines(s, arm='fixed'):
    d = f'res3/{arm}/s{s}'
    if arm == 'online':     # SAC and CREST-C first; DSAC-N as a separate pair of jobs once its checkpoint exists
        return [f'{d}/done_prep.txt | logs/eval_online_s{s}_prep.log | runs_online/sac_s{s}.pkl,runs_online/crest_s{s}.pkl,res3/reactive_params.json | '
                f'{PY} suite3.py --seed {s} --arm online --part prep --agents sac,crest',
                f'{d}/done_main.txt | logs/eval_online_s{s}_main.log | {d}/done_prep.txt | {PY} suite3.py --seed {s} --arm online --part main --agents sac,crest',
                f'{d}/done_prep_dsacn.txt | logs/eval_online_s{s}_prep_dsacn.log | runs_online/dsacn_s{s}.pkl,{d}/done_main.txt | '
                f'{PY} suite3.py --seed {s} --arm online --part prep --agents dsacn --tag _dsacn',
                f'{d}/done_main_dsacn.txt | logs/eval_online_s{s}_main_dsacn.log | {d}/done_prep_dsacn.txt | '
                f'{PY} suite3.py --seed {s} --arm online --part main --agents dsacn --tag _dsacn']
    req = ','.join([f'runs/{a}_s{s}.pkl' for a in MAINALG] + ['res3/reactive_params.json'])
    R = [f'{d}/done_prep.txt | logs/eval_{arm}_s{s}_prep.log | {req} | {PY} suite3.py --seed {s} --arm {arm} --part prep']
    for p in PARTS:
        R.append(f'{d}/done_{p}.txt | logs/eval_{arm}_s{s}_{p}.log | {d}/done_prep.txt | {PY} suite3.py --seed {s} --arm {arm} --part {p}')
    R.append(f'{d}/done_critic.txt | logs/eval_{arm}_s{s}_critic.log | {d}/done_prep.txt | {PY} critic_check.py --seed {s} --arm {arm}')
    return R


L = ['# ---- non-learning reactive controller: parameters selected on nominal validation districts',
     f'res3/reactive_params.json | logs/tune_reactive.log | - | {PY} tune_reactive.py']
# seeds 0-2 of SAC, TD3, DSAC-N and DSAC-CVaR are the checkpoints of the original submission (fixed-reserve training)
for s in range(3):
    L.append(f'# ---- seed {s}')
    for a in ['crest', 'crestret']:
        L.append(train_line(a, s, 'fixed'))
    if s == 0:
        L += eval_lines(0)
        L.append(f'analysis/qual.npz | logs/qual3.log | res3/fixed/s0/done_prep.txt | {PY} qual3.py')
for s in (1, 2):
    L += eval_lines(s)
for s in range(3, 10):
    L.append(f'# ---- seed {s}')
    for a in MAINALG:
        L.append(train_line(a, s, 'fixed'))
    L += eval_lines(s)
L.append('# ---- PPO reference: submitted implementation at four times the budget (seeds 0-2 at the main budget are in runs_legacy)')
for s in (0, 1):
    L.append(f'runs_legacy/ppo100k_s{s}.pkl | logs/train_legacy_ppo100k_s{s}.log | - | {PY} train.py --algo ppo --seed {s} --steps 100000 '
             f'--ppo_variant legacy --reserve fixed --out runs_legacy/ppo100k_tmp_s{s} && mv runs_legacy/ppo100k_tmp_s{s}/ppo_s{s}.pkl runs_legacy/ppo100k_s{s}.pkl')
L.append(f'res3/ppo/done.txt | logs/eval_ppo.log | runs_legacy/ppo100k_s0.pkl,runs_legacy/ppo100k_s1.pkl | {PY} eval_ppo.py')
L.append('# ---- online-trained arm, five seeds, one seed at a time (seeds 0-2 of CREST-C are the checkpoints of the original submission)')
for s in range(5):
    L.append(train_line('sac', s, 'online'))
    if s >= 3:
        L.append(train_line('crest', s, 'online'))
    L.append(train_line('dsacn', s, 'online'))
    L += eval_lines(s, 'online')
os.makedirs('jobs', exist_ok=True)
open('jobs/queue.txt', 'w').write('\n'.join(L) + '\n')
print(len([l for l in L if not l.startswith('#')]), 'jobs written to jobs/queue.txt')
