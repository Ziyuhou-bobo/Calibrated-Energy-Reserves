"""MATLAB-like figure style and the study palette (revision figures are produced with matplotlib).

Boxed axes, inward ticks on all four sides, light major grid, Helvetica-metric font (Liberation Sans),
legends placed below the axes outside the plotting area, and distinct colour + marker + line style per
policy so that series remain distinguishable in greyscale.  Figures are saved as vector PDF and EPS
(EPS has no transparency, so shaded bands use pre-blended solid colours) plus a PNG preview.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

CM = 1 / 2.54
OUTDIR = os.environ.get('FIGOUT', 'figs/out')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'FreeSans', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'axes.edgecolor': '#262626', 'axes.labelcolor': '#1a1a1a',
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.color': '#262626', 'ytick.color': '#262626',
    'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': 0.5, 'grid.linestyle': '-',
    'axes.axisbelow': True, 'legend.frameon': False, 'legend.handlelength': 2.6, 'legend.columnspacing': 1.4,
    'lines.linewidth': 1.2, 'lines.markersize': 4.2, 'errorbar.capsize': 1.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600, 'figure.dpi': 150,
    'axes.titlepad': 4.0, 'mathtext.default': 'regular',
})

# ---------------------------------------------------------------- palette (study-specific, non-default)
POL = {
    'sac':      dict(lab='SAC',       col=(0.35, 0.44, 0.66), mk='s', ls='-'),
    'td3':      dict(lab='TD3',       col=(0.50, 0.56, 0.22), mk='D', ls=(0, (6, 2))),
    'dsacn':    dict(lab='DSAC-N',    col=(0.82, 0.60, 0.20), mk='v', ls=(0, (4, 1.5, 1, 1.5))),
    'dsaccvar': dict(lab='DSAC-CVaR', col=(0.16, 0.52, 0.55), mk='o', ls=(0, (1.5, 1.5))),
    'crestret': dict(lab='CREST-R',   col=(0.86, 0.47, 0.62), mk='^', ls=(0, (7, 1.5, 1.5, 1.5, 1.5, 1.5))),
    'crest':    dict(lab='CREST-C',   col=(0.78, 0.16, 0.14), mk='p', ls='-'),
    'reactive': dict(lab='reactive rule', col=(0.22, 0.22, 0.22), mk='X', ls=(0, (3, 1.2))),
    'ppo':      dict(lab='PPO',       col=(0.58, 0.38, 0.58), mk='h', ls=(0, (2, 1))),
}
GUARD = {
    'std':    dict(lab=r'fixed 1.2$\times$', col=(0.55, 0.55, 0.55), hatch=''),
    'std15':  dict(lab=r'fixed 1.5$\times$', col=(0.70, 0.56, 0.38), hatch='////'),
    'static': dict(lab='split conformal',     col=(0.24, 0.40, 0.64), hatch='....'),
    'online': dict(lab='online conformal',    col=(0.78, 0.16, 0.14), hatch=''),
}
COND = ['nominal', 'wind', 'sensor', 'traffic', 'energy', 'compound']
CONDLAB = dict(nominal='nominal', wind='wind', sensor='sensor', traffic='traffic', energy='energy', compound='compound')


def light(c, f=0.75):
    """Blend colour c with white (f = weight of white); solid colour, safe for EPS."""
    c = np.array(to_rgb(c)); return tuple(c + (1 - c) * f)


def style(ax, grid_x=True):
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color('#262626')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(True, axis='y'); ax.grid(grid_x, axis='x')


def panel_title(ax, s):
    ax.set_title(s, loc='center', fontweight='normal')


def legend_below(fig, handles, labels, ncol, y=0.0, fontsize=7.5):
    """Legend centred below all axes, outside the plotting areas."""
    return fig.legend(handles, labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False,
                      fontsize=fontsize, handlelength=2.6, columnspacing=1.6)


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ('pdf', 'eps'):
        fig.savefig(f'{OUTDIR}/{name}.{ext}', format=ext)
    fig.savefig(f'{OUTDIR}/{name}.png', dpi=200)
    plt.close(fig)
    print('saved', name)


def errorbar_ci(ax, x, y, lo, hi, **kw):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
    return ax.errorbar(x, y, yerr=yerr, **kw)
