import re
s = open('pa/main.tex').read()
def inl(m):
    p = m.group(1); p = p if p.endswith('.tex') else p + '.tex'
    return open('pa/' + p).read()
s = re.sub(r'\\input\{([^}]+)\}', inl, s)
s = s.replace("\\graphicspath{{figs/}{../figs/outA/}}\n", "")
blocks = []
for env in ('figure', 'table'):
    for m in re.finditer(r'\\begin\{%s\}.*?\\end\{%s\}\n?' % (env, env), s, flags=re.S):
        blocks.append((m.group(0), re.search(r'\\label\{([^}]+)\}', m.group(0)).group(1), env))
for blk, lab, env in blocks:
    s = s.replace(blk, '')
s = re.sub(r'\n{3,}', '\n\n', s).replace('\\clearpage\n\n', '').replace('\\clearpage\n', '')
pos = {}
for _, lab, _ in blocks:
    h = re.search(r'\\ref\{%s\}' % re.escape(lab), s); pos[lab] = h.end() if h else len(s)
for blk, lab, env in sorted(blocks, key=lambda b: -pos[b[1]]):
    b = re.sub(r'\\begin\{%s\}\[[^\]]*\]' % env, r'\\begin{%s}[!htbp]' % env, blk, count=1)
    b = b if b.endswith('\n') else b + '\n'
    nxt = s.find('\n\n', pos[lab]); nxt = len(s) if nxt < 0 else nxt
    s = s[:nxt + 1] + '\n' + b + s[nxt + 1:]
open('/tmp/build_main.tex', 'w').write(s)
print('floats placed:', len(blocks))
