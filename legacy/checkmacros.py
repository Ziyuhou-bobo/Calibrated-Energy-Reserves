import re, sys, glob
defined = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open('pa/data/macros.tex').read()))
used = set()
for f in glob.glob('pa/sec_*.tex') + glob.glob('pa/main.tex') + glob.glob('pa/tab_*.tex'):
    used |= set(re.findall(r'\\([RCSAPD][A-Za-z]{4,})\b', open(f).read()))
miss = sorted(u for u in used if u not in defined)
print('used', len(used), 'missing', len(miss))
print('\n'.join(miss[:40]))
