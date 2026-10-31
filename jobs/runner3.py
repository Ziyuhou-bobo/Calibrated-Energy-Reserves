"""Resumable job queue with dependencies.

Queue line formats (fields separated by '|'):
    OUTPUT | LOG | shell command
    OUTPUT | LOG | REQ1,REQ2,... | shell command      (job waits until every REQ path exists)
Jobs are started in file order whenever a worker is free and their requirements exist; a job whose output
exists is skipped.  The number of workers is read from workers.txt at every poll.  Commands that are already
running (for example, started by a previous runner) are detected through /proc and adopted rather than
restarted.  The runner keeps polling for new lines until a file STOP exists and nothing is left to do.
"""
import os, sys, time, subprocess

qfile = os.path.abspath(sys.argv[1]); qdir = os.path.dirname(qfile)
os.chdir(sys.argv[2] if len(sys.argv) > 2 else '.')
env = dict(os.environ, XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
running, adopted, failed, done_set = {}, {}, set(), set()
NCPU = os.cpu_count() or 1
cores = {}          # output -> core the job is pinned to


def pin(pid, core):
    """Pin a process and all its descendants (all threads) to one core."""
    pids = [pid]
    try:
        kids = subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True).stdout.split()
        pids += [int(k) for k in kids]
    except Exception:
        pass
    for p in pids:
        subprocess.run(['taskset', '-a', '-p', '-c', str(core), str(p)], capture_output=True)


def free_core():
    used = list(cores.values())
    for c in range(NCPU):
        if c not in used:
            return c
    return min(range(NCPU), key=used.count)


def log(*a):
    with open(os.path.join(qdir, 'runner.log'), 'a') as f:
        f.write(time.strftime('%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a) + '\n')


def proc_cmdlines():
    out = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                out[int(p)] = open(f'/proc/{p}/cmdline', 'rb').read().split(b'\0')
            except Exception:
                pass
    return out


def parse():
    jobs = []
    for line in open(qfile):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [x.strip() for x in line.split('|')]
        if len(parts) == 3:
            out, lg, cmd = parts; req = []
        else:
            out, lg, rq = parts[:3]; cmd = '|'.join(parts[3:]).strip()
            req = [] if rq in ('', '-') else [x.strip() for x in rq.split(',') if x.strip()]
        jobs.append((out, lg, req, cmd))
    return jobs


first = True
while True:
    try:
        W = int(open(os.path.join(qdir, 'workers.txt')).read().strip())
    except Exception:
        W = 2
    jobs = parse()
    if first:   # adopt identical commands that are already running
        cl = proc_cmdlines()
        for out, lg, req, cmd in jobs:
            if os.path.exists(out):
                continue
            for pid, args in cl.items():
                if pid == os.getpid():
                    continue
                if any(a.decode(errors='ignore') == cmd for a in args):
                    adopted[out] = pid; c = free_core(); cores[out] = c; pin(pid, c); log('adopted', out, pid, 'core', c); break
        first = False
    for out, p in list(running.items()):
        rc = p.poll()
        if rc is not None:
            del running[out]; cores.pop(out, None)
            if rc == 0 and os.path.exists(out):
                done_set.add(out); log('done', out)
            else:
                failed.add(out); log('FAILED rc=%s' % rc, out)
    for out, pid in list(adopted.items()):
        if not os.path.exists(f'/proc/{pid}'):
            del adopted[out]; cores.pop(out, None)
            if os.path.exists(out):
                done_set.add(out); log('done (adopted)', out)
            else:
                failed.add(out); log('FAILED (adopted)', out)
    busy = set(running) | set(adopted)
    pending = [j for j in jobs if not os.path.exists(j[0]) and j[0] not in busy and j[0] not in failed]
    ready = [j for j in pending if all(os.path.exists(r) for r in j[2])]
    while len(running) + len(adopted) < W and ready:
        out, lg, req, cmd = ready.pop(0)
        os.makedirs(os.path.dirname(lg) or '.', exist_ok=True)
        fh = open(lg, 'w')
        c = free_core(); cores[out] = c
        running[out] = subprocess.Popen(f'taskset -c {c} sh -c ' + "'" + cmd.replace("'", "'\\''") + "'", shell=True, stdout=fh,
                                        stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log('start', out, 'core', c, '::', cmd)
    with open(os.path.join(qdir, 'status.txt'), 'w') as f:
        f.write('%s running %d pending %d (ready %d) failed %d done %d\n' % (
            time.strftime('%H:%M:%S'), len(running) + len(adopted), len(pending), len(ready), len(failed), len(done_set)))
        for o in list(running) + list(adopted):
            f.write('  R ' + o + '\n')
        for o in sorted(failed):
            f.write('  F ' + o + '\n')
    if not running and not adopted and not pending and os.path.exists(os.path.join(qdir, 'STOP')):
        log('exit'); break
    time.sleep(15)
