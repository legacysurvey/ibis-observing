#!/usr/bin/env python
"""Coverage-completeness maps of the DESI Run 2 medium-band imaging, one panel per filter.

At each point of the sky the map shows (tiles DONE covering it) / (program tiles
covering it), from 0 (nothing observed) to 1 (every planned exposure of that
point taken).  The five rows are the bands MB1-MB5; the columns are the NGC and
SGC.  CFHT images MB1-MB2 and DECam MB3-MB5 in the POLU+IBIS sky; the DECam
M411/M438 exposures are added to the CFHT fraction of MB1/MB2 (clipped at 1).
The HSC-Niji+VST rectangles are drawn in grey: there is no status file for them yet.

Program tiles are those with IN_IBIS=1 and IN_HSC=0 (DECam: wide tiles only),
as in stripe_completeness.py.  The stripes are outlined by their unique
2.8-deg declination bands and labelled.  The number at the right-hand end of
each stripe is the mean of the map over that band (where there are program tiles).

  python plot_stripe_completeness.py --cfht-tiles ~/cfht-observing/obstatus/cfht-tiles.ecsv
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFHT = os.path.join(HERE, '..', '..', 'cfht-observing', 'obstatus', 'cfht-tiles.ecsv')
CELL = 0.1
# plotting windows in x (= RA for the NGC, RA wrapped to +-180 for the SGC) and Dec
CAPS = {'NGC': dict(x=(108.0, 270.0), dec=(-9.5, 16.5)), 'SGC': dict(x=(-68.0, 74.0), dec=(-19.5, 6.5))}
HALF = {'CFHT': (0.5, 0.5), 'DECam': (1.05, 0.95)}     # field half-sizes (RA on the sky, Dec), as footprint.py
BANDS = [('MB1', 'M4112', 'M411'), ('MB2', 'M4376', 'M438'), ('MB3', None, 'M464'), ('MB4', None, 'M490'), ('MB5', None, 'M517')]
CMAP = LinearSegmentedColormap.from_list('completeness', ['#dfe9f5', '#9dbfe3', '#4f8fcf', '#1f5fae', '#0b2f6b'])
NIJI_GREY = '#d9d9d9'; INK = '#333333'


def capx(ra, cap):
    ra = np.asarray(ra, float) % 360
    return ra if cap == 'NGC' else (ra + 180) % 360 - 180


def is_ngc(ra):
    ra = np.asarray(ra, float) % 360
    return (ra > 90) & (ra < 270)


def raster(ra, dec, half, cap):
    """Number of tiles covering each cell of the cap's grid."""
    (x0, x1), (d0, d1) = CAPS[cap]['x'], CAPS[cap]['dec']
    nx, ny = int(round((x1 - x0) / CELL)), int(round((d1 - d0) / CELL))
    g = np.zeros((ny, nx), np.int32)
    x = capx(ra, cap); dec = np.asarray(dec, float)
    hx = half[0] / np.cos(np.radians(dec))
    i0 = np.clip(np.round((x - hx - x0) / CELL).astype(int), 0, nx); i1 = np.clip(np.round((x + hx - x0) / CELL).astype(int), 0, nx)
    j0 = np.clip(np.round((dec - half[1] - d0) / CELL).astype(int), 0, ny); j1 = np.clip(np.round((dec + half[1] - d0) / CELL).astype(int), 0, ny)
    for a, b, c, d in zip(i0, i1, j0, j1):
        g[c:d, a:b] += 1
    return g


def rect_x(ra_min, ra_max, cap):
    """x-range of an RA interval (which may wrap through 0) in the cap's coordinate."""
    a, b = float(capx(ra_min, cap)), float(capx(ra_max, cap))
    return a, b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ibis-tiles', default=os.path.join(HERE, 'ibis-tiles.ecsv'))
    p.add_argument('--cfht-tiles', default=DEFAULT_CFHT)
    p.add_argument('--stripes', default=os.path.join(HERE, 'desi2-stripes.ecsv'))
    p.add_argument('--regions', default=os.path.join(HERE, 'footprint-regions.ecsv'))
    p.add_argument('-o', '--out', default=os.path.join(HERE, 'stripe-completeness.pdf'))
    a = p.parse_args()

    st = Table.read(a.stripes); st = st[np.array([str(n) != 'extended' for n in st['NOTE']])]
    rg = Table.read(a.regions); rg = rg[rg['KIND'] == 'niji']
    tiles = {}
    t = Table.read(a.ibis_tiles)
    sel = np.array([o.startswith('IBIS_wide') for o in t['OBJECT']]) & (np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)
    tiles['DECam'] = t[sel]
    if os.path.exists(a.cfht_tiles):
        t = Table.read(a.cfht_tiles)
        tiles['CFHT'] = t[(np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)]
    else:
        print('No CFHT tile file at %s: MB1/MB2 show DECam only' % a.cfht_tiles)

    def fraction(cam, filt, cap):
        t = tiles[cam]; m = (np.asarray(t['FILTER']) == filt) & (is_ngc(t['RA']) == (cap == 'NGC'))
        plan = raster(t['RA'][m], t['DEC'][m], HALF[cam], cap)
        md = m & (np.asarray(t['DONE']) > 0)
        done = raster(t['RA'][md], t['DEC'][md], HALF[cam], cap)
        return plan, np.where(plan > 0, done / np.maximum(plan, 1), 0.0)

    widths = [CAPS[c]['x'][1] - CAPS[c]['x'][0] for c in CAPS]
    height = CAPS['NGC']['dec'][1] - CAPS['NGC']['dec'][0]
    figw = 14.0; panelh = (figw - 1.2) * height / sum(widths)
    fig, axes = plt.subplots(len(BANDS), 2, figsize=(figw, len(BANDS) * (panelh + 0.42) + 1.0),
                             gridspec_kw=dict(width_ratios=widths, wspace=0.07, hspace=0.38))
    for row, (band, fc, fd) in enumerate(BANDS):
        primary = 'CFHT' if (fc and 'CFHT' in tiles) else 'DECam'
        ntile = []
        for col, cap in enumerate(CAPS):
            ax = axes[row, col]; (x0, x1), (d0, d1) = CAPS[cap]['x'], CAPS[cap]['dec']
            if primary == 'CFHT':
                plan, frac = fraction('CFHT', fc, cap)
                frac = np.clip(frac + fraction('DECam', fd, cap)[1], 0, 1)
            else:
                plan, frac = fraction('DECam', fd, cap)
            for r in rg:                       # HSC-Niji+VST sky, under the map
                if r['NAME'].startswith(cap):
                    xa, xb = rect_x(r['RA_MIN'], r['RA_MAX'], cap)
                    ax.add_patch(Rectangle((xa, r['DEC_MIN']), xb - xa, r['DEC_MAX'] - r['DEC_MIN'], fc=NIJI_GREY, ec='none', zorder=1))
            im = ax.imshow(np.ma.masked_where(plan == 0, frac), origin='lower', extent=(x0, x1, d0, d1), cmap=CMAP, vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest', zorder=2)
            for s in st:                       # stripe outlines and labels
                if not s['STRIPE'].startswith(cap): continue
                xa, xb = rect_x(s['RA_MIN'], s['RA_MAX'], cap)
                ax.add_patch(Rectangle((xa, s['DEC_CENTER'] - 1.4), xb - xa, 2.8, fc='none', ec=INK, lw=0.5, zorder=3))
                ax.text(x1 - 1.0, s['DEC_CENTER'], s['STRIPE'], ha='left', va='center', fontsize=6.5, color=INK, zorder=4)
                # mean of the map over the stripe's unique band (the part with program tiles)
                ny, nx = plan.shape
                xc = x0 + (np.arange(nx) + 0.5) * CELL; dc = d0 + (np.arange(ny) + 0.5) * CELL
                inside = ((dc > s['DEC_CENTER'] - 1.4) & (dc < s['DEC_CENTER'] + 1.4))[:, None] & ((xc > xa) & (xc < xb))[None, :] & (plan > 0)
                if inside.any() and s['POLU_IBIS'] == 1:     # no number for a stripe CFHT/DECam do not image
                    ax.text(x0 + 1.0, s['DEC_CENTER'], '%.0f%%' % (100 * frac[inside].mean()), ha='right', va='center', fontsize=6.5, color=INK, zorder=4)
            ax.set_xlim(x1, x0); ax.set_ylim(d0, d1)          # RA increasing to the left
            ticks = np.arange(np.ceil(x0 / 20) * 20, x1, 20)
            ax.set_xticks(ticks); ax.set_xticklabels(['%d' % (v % 360) for v in ticks])
            ax.tick_params(labelsize=8, length=3, color='#888888')
            for sp in ax.spines.values(): sp.set_color('#aaaaaa'); sp.set_linewidth(0.6)
            if col == 0: ax.set_ylabel('Dec (deg)', fontsize=9)
            if row == len(BANDS) - 1: ax.set_xlabel('RA (deg)', fontsize=9)
        parts = []
        for cam, f in (('CFHT', fc), ('DECam', fd)):
            if f and cam in tiles:
                t = tiles[cam]; m = np.asarray(t['FILTER']) == f; d = int((np.asarray(t['DONE'])[m] > 0).sum())
                parts.append('%s %s: %d of %d tiles (%.1f%%)' % (cam, f, d, m.sum(), 100.0 * d / m.sum()))
        axes[row, 0].set_title(band + '    ' + ';   '.join(parts), loc='left', fontsize=10, color='black')
    axes[0, 0].text(1.0, 1.04, 'NGC', transform=axes[0, 0].transAxes, ha='right', va='bottom', fontsize=10, color=INK)
    axes[0, 1].text(1.0, 1.04, 'SGC', transform=axes[0, 1].transAxes, ha='right', va='bottom', fontsize=10, color=INK)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.925, bottom=0.115)
    cax = fig.add_axes([0.30, 0.045, 0.34, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=8)
    cb.set_label('fraction of planned exposures completed at each point', fontsize=9)
    fig.legend(handles=[Patch(fc=NIJI_GREY, ec='none', label='HSC-Niji + VST sky (no status yet)'), Patch(fc='none', ec=INK, lw=0.5, label='stripe (unique 2.8-deg band)')],
               loc='lower right', bbox_to_anchor=(0.985, 0.028), frameon=False, fontsize=8.5, ncol=1)
    fig.suptitle('DESI Run 2 medium-band imaging: completeness by stripe', x=0.05, y=0.985, ha='left', fontsize=12)
    fig.savefig(a.out); print('wrote', a.out)
    if a.out.endswith('.pdf'):
        fig.savefig(a.out[:-4] + '.png', dpi=110); print('wrote', a.out[:-4] + '.png')


if __name__ == '__main__':
    main()
