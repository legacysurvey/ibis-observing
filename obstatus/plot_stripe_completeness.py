#!/usr/bin/env python
"""Coverage-completeness maps of the DESI Run 2 medium-band imaging, one panel per filter.

At each point of the sky the map shows (tiles DONE covering it) / (program tiles
covering it), from 0 (nothing observed) to 1 (every planned exposure of that
point taken).  The five rows are the bands MB1-MB5; the columns are the NGC and
SGC.  CFHT images MB1-MB2 and DECam MB3-MB5 in the POLU+IBIS sky; the DECam
M411/M438 exposures are added to the CFHT fraction of MB1/MB2 (clipped at 1).
Inside the HSC-Niji+VST rectangles MB1-MB4 show HSC-Niji: every exposure is taken
through the MBQ1 quadrant filter, so each of the four filters (413, 439, 465, 490)
sees about a quarter of the focal plane.  The map is (throughput-weighted usable
shots) / (throughput-weighted planned shots) in that filter, from the survey
status table and the per-filter throughput maps of the Niji
wide-observing-strategy repository (--niji-repo).  Shots count as usable to the
loose seeing limit of 2.0 arcsec (USABLE_LOOSE) unless --niji-strict asks for the
1.3 arcsec limit (USABLE).  MB5 there will come from VST, which has no status yet,
so that area is left empty in the MB5 row.  The HSC-Niji+VST area is outlined in
every row; inside it only HSC (and, later, VST) coverage is shown, never the
CFHT or DECam tiles that overlap its edge.  Parts of a stripe that nobody images
(none since 2026-11-20, when the stripe table was trimmed to the imaged sky) would be hatched.

Program tiles are those with IN_IBIS=1 and IN_HSC=0 (DECam: wide tiles only),
as in stripe_completeness.py.  The stripes are outlined by their unique
2.8-deg declination bands and labelled.  The number at the right-hand end of
each stripe is the mean of the map over that band (where there are program tiles).

  python plot_stripe_completeness.py --cfht-tiles ~/cfht-observing/obstatus/cfht-tiles.ecsv
"""
import argparse, csv, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['hatch.color'] = '#777777'; matplotlib.rcParams['hatch.linewidth'] = 0.4
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFHT = os.path.join(HERE, '..', '..', 'cfht-observing', 'obstatus', 'cfht-tiles.ecsv')
CELL = 0.1
# plotting windows in x (= RA for the NGC, RA wrapped to +-180 for the SGC) and Dec
CAPS = {'NGC': dict(x=(108.0, 270.0), dec=(-12.0, 16.5)), 'SGC': dict(x=(-68.0, 74.0), dec=(-22.0, 6.5))}
HALF = {'CFHT': (0.5, 0.5), 'DECam': (1.05, 0.95)}     # field half-sizes (RA on the sky, Dec), as footprint.py
BANDS = [('MB1', 'M4112', 'M411', '413'), ('MB2', 'M4376', 'M438', '439'), ('MB3', None, 'M464', '465'), ('MB4', None, 'M490', '490'),
         ('MB5', None, 'M517', None)]       # band, CFHT filter, DECam filter, HSC MBQ1 sub-filter
DEFAULT_NIJI = os.path.join(os.path.expanduser('~'), 'Imaging', 'wide-observing-strategy')
NIJI_MIN_PLAN = 0.5                  # effective planned exposures below which a cell is not drawn for HSC
CMAP = LinearSegmentedColormap.from_list('completeness', ['#dfe9f5', '#9dbfe3', '#4f8fcf', '#1f5fae', '#0b2f6b'])
NIJI_LINE = '#d9730d'; INK = '#333333'


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


class Niji:
    """HSC-Niji planned and usable coverage per MBQ1 sub-filter, on the cap grids.

    The coverage of a sky point by one shot is the relative throughput of the
    sub-filter at the point's position in the focal plane (the map file written
    by skytiling's flats_to_throughput; ROT follows skytiling's convention).
    """

    def __init__(self, repo, usable_column='USABLE', throughput=None):
        from astropy.io import fits
        self.status = os.path.join(repo, 'survey_status.csv')
        thru = throughput or os.path.join(repo, 'wide_dithering', 'hsc_mbq1_throughput_8p18_ccd44dead.fits')
        rows = list(csv.DictReader(open(self.status)))
        self.ra = np.array([float(r['RA']) for r in rows]); self.dec = np.array([float(r['DEC']) for r in rows])
        self.rot = np.radians([float(r['ROT']) for r in rows])
        self.usable = np.array([r[usable_column] == '1' for r in rows])
        with fits.open(thru) as h:
            ph = h[0].header
            self.filters = [str(ph['FILT%d' % i]) for i in range(ph['NFILT'])]
            self.res = float(ph['RES']) / 60.0; self.half = float(ph['EXTENT']) / 60.0
            self.maps = np.stack([np.nan_to_num(h['THRU_' + f].data.astype(np.float32)) for f in self.filters])
        self.n = self.maps.shape[1]; self._cache = {}

    def grids(self, cap):
        """(plan, done) arrays of shape (nfilter, ny, nx): effective exposures per cell."""
        if cap in self._cache: return self._cache[cap]
        (x0, x1), (d0, d1) = CAPS[cap]['x'], CAPS[cap]['dec']
        nx, ny = int(round((x1 - x0) / CELL)), int(round((d1 - d0) / CELL))
        plan = np.zeros((len(self.filters), ny, nx), np.float32); done = np.zeros_like(plan)
        x = capx(self.ra, cap); R = self.half * np.sqrt(2)
        sel = np.flatnonzero((is_ngc(self.ra) == (cap == 'NGC')) & (self.dec > d0 - R) & (self.dec < d1 + R))
        for k in sel:
            cd = np.cos(np.radians(self.dec[k])); hx = R / cd
            i0, i1 = max(int((x[k] - hx - x0) / CELL), 0), min(int((x[k] + hx - x0) / CELL) + 1, nx)
            j0, j1 = max(int((self.dec[k] - R - d0) / CELL), 0), min(int((self.dec[k] + R - d0) / CELL) + 1, ny)
            if i1 <= i0 or j1 <= j0: continue
            ra = np.radians(x0 + (np.arange(i0, i1) + 0.5) * CELL)[None, :]; de = np.radians(d0 + (np.arange(j0, j1) + 0.5) * CELL)[:, None]
            ra0, de0 = np.radians(x[k]), np.radians(self.dec[k])
            cosc = np.sin(de0) * np.sin(de) + np.cos(de0) * np.cos(de) * np.cos(ra - ra0)        # gnomonic projection about the shot
            lx = np.degrees(np.cos(de) * np.sin(ra - ra0) / cosc)                                # east
            ly = np.degrees((np.cos(de0) * np.sin(de) - np.sin(de0) * np.cos(de) * np.cos(ra - ra0)) / cosc)   # north
            c, s = np.cos(self.rot[k]), np.sin(self.rot[k])
            ix = np.floor((lx * c + ly * s + self.half) / self.res).astype(int); iy = np.floor((-lx * s + ly * c + self.half) / self.res).astype(int)
            ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
            v = np.where(ok[None], self.maps[:, np.clip(iy, 0, self.n - 1), np.clip(ix, 0, self.n - 1)], 0.0)
            plan[:, j0:j1, i0:i1] += v
            if self.usable[k]: done[:, j0:j1, i0:i1] += v
        self._cache[cap] = (plan, done)
        return plan, done


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
    p.add_argument('--niji-repo', default=DEFAULT_NIJI, help='clone of hsc-niji-survey/wide-observing-strategy (survey_status.csv and throughput maps)')
    p.add_argument('--niji-strict', action='store_true', help='count only HSC shots with seeing <= 1.3 arcsec (USABLE); default is the loose 2.0 arcsec limit (USABLE_LOOSE)')
    p.add_argument('-o', '--out', default=os.path.join(HERE, 'stripe-completeness.pdf'))
    a = p.parse_args()

    st = Table.read(a.stripes)
    rgall = Table.read(a.regions); rg = rgall[rgall['KIND'] == 'niji']; dropped = rgall[rgall['KIND'] == 'dropped']
    tiles = {}
    t = Table.read(a.ibis_tiles)
    sel = np.array([o.startswith('IBIS_wide') for o in t['OBJECT']]) & (np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)
    tiles['DECam'] = t[sel]
    if os.path.exists(a.cfht_tiles):
        t = Table.read(a.cfht_tiles)
        tiles['CFHT'] = t[(np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)]
    else:
        print('No CFHT tile file at %s: MB1/MB2 show DECam only' % a.cfht_tiles)
    niji = None
    if os.path.exists(os.path.join(a.niji_repo, 'survey_status.csv')):
        niji = Niji(a.niji_repo, 'USABLE' if a.niji_strict else 'USABLE_LOOSE')
    else:
        print('No Niji survey_status.csv under %s: HSC not shown (use --niji-repo)' % a.niji_repo)

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
    any_noimg = False
    for row, (band, fc, fd, fh) in enumerate(BANDS):
        primary = 'CFHT' if (fc and 'CFHT' in tiles) else 'DECam'
        ntile = []
        for col, cap in enumerate(CAPS):
            ax = axes[row, col]; (x0, x1), (d0, d1) = CAPS[cap]['x'], CAPS[cap]['dec']
            if primary == 'CFHT':
                plan, frac = fraction('CFHT', fc, cap)
                frac = np.clip(frac + fraction('DECam', fd, cap)[1], 0, 1)
            else:
                plan, frac = fraction('DECam', fd, cap)
            ny, nx = plan.shape
            xc = x0 + (np.arange(nx) + 0.5) * CELL; dc = d0 + (np.arange(ny) + 0.5) * CELL
            plan = plan > 0
            def cells(xa, xb, da, db):
                return ((dc > da) & (dc < db))[:, None] & ((xc > xa) & (xc < xb))[None, :]
            # the stripes' unique bands, the HSC-Niji+VST area within them, and the parts of the stripes nobody images
            stripes = np.zeros(plan.shape, bool); nijimask = np.zeros(plan.shape, bool); noimg = np.zeros(plan.shape, bool)
            for s in st:
                if s['STRIPE'].startswith(cap):
                    sband = cells(*rect_x(s['RA_MIN'], s['RA_MAX'], cap), s['DEC_CENTER'] - 1.4, s['DEC_CENTER'] + 1.4)
                    stripes |= sband
                    if s['POLU_IBIS'] != 1: noimg |= sband          # e.g. the ends of SGC-1 (its HSC part is removed below)
            for r in dropped:
                if r['NAME'].startswith(cap): noimg |= cells(*rect_x(r['RA_MIN'], r['RA_MAX'], cap), r['DEC_MIN'], r['DEC_MAX']) & stripes
            for r in rg:
                if r['NAME'].startswith(cap): nijimask |= cells(*rect_x(r['RA_MIN'], r['RA_MAX'], cap), r['DEC_MIN'], r['DEC_MAX'])
            nijimask &= stripes; noimg &= ~nijimask
            plan &= ~nijimask                  # inside the HSC-Niji+VST area only HSC (and, one day, VST) is shown
            for r in rg:
                if r['NAME'].startswith(cap):
                    xa, xb = rect_x(r['RA_MIN'], r['RA_MAX'], cap)
                    if niji is not None and fh:
                        hp, hd = niji.grids(cap); k = niji.filters.index(fh)
                        inr = cells(xa, xb, r['DEC_MIN'], r['DEC_MAX']) & nijimask & (hp[k] >= NIJI_MIN_PLAN)
                        frac = np.where(inr, np.clip(hd[k] / np.maximum(hp[k], 1e-6), 0, 1), frac); plan = plan | inr
            im = ax.imshow(np.ma.masked_where(~plan, frac), origin='lower', extent=(x0, x1, d0, d1), cmap=CMAP, vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest', zorder=2)
            if noimg.any():
                any_noimg = True
                ax.contourf(xc, dc, noimg.astype(float), levels=[0.5, 1.5], colors='none', hatches=['//////'], zorder=2.5)
            if nijimask.any():
                ax.contour(xc, dc, nijimask.astype(float), levels=[0.5], colors=[NIJI_LINE], linewidths=1.1, zorder=5)
            for s in st:                       # stripe outlines and labels
                if not s['STRIPE'].startswith(cap): continue
                xa, xb = rect_x(s['RA_MIN'], s['RA_MAX'], cap)
                ax.add_patch(Rectangle((xa, s['DEC_CENTER'] - 1.4), xb - xa, 2.8, fc='none', ec=INK, lw=0.5, zorder=3))
                ax.text(x1 - 1.0, s['DEC_CENTER'], s['STRIPE'], ha='left', va='center', fontsize=6.5, color=INK, zorder=4)
                # mean of the map over the stripe's unique band (the part with program tiles)
                inside = ((dc > s['DEC_CENTER'] - 1.4) & (dc < s['DEC_CENTER'] + 1.4))[:, None] & ((xc > xa) & (xc < xb))[None, :] & plan
                if inside.sum() * CELL ** 2 > 20 and (s['POLU_IBIS'] == 1 or (niji is not None and fh)):   # no number where nothing with a status is planned
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
        if niji is not None and fh:
            axes[row, 1].set_title('HSC-Niji MBQ1-%s: %d of %d shots (%.1f%%), seeing <= %s arcsec' % (fh, niji.usable.sum(), len(niji.usable), 100.0 * niji.usable.mean(), '1.3' if a.niji_strict else '2.0'),
                                   loc='left', fontsize=10, color='black')
        elif niji is not None:
            axes[row, 1].set_title('VST: no status yet', loc='left', fontsize=10, color='black')
        axes[row, 0].set_title(band + '    ' + ';   '.join(parts), loc='left', fontsize=10, color='black')
    axes[0, 0].text(1.0, 1.04, 'NGC', transform=axes[0, 0].transAxes, ha='right', va='bottom', fontsize=10, color=INK)
    axes[0, 1].text(1.0, 1.04, 'SGC', transform=axes[0, 1].transAxes, ha='right', va='bottom', fontsize=10, color=INK)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.925, bottom=0.115)
    cax = fig.add_axes([0.30, 0.045, 0.34, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=8)
    cb.set_label('fraction of planned exposures completed at each point', fontsize=9)
    handles = [Line2D([], [], color=NIJI_LINE, lw=1.1, label='HSC-Niji + VST area (MB5 from VST: no status yet)'), Patch(fc='none', ec=INK, lw=0.5, label='stripe (unique 2.8-deg band)')]
    if any_noimg:
        handles.append(Patch(fc='none', ec='#777777', lw=0.0, hatch='//////', label='part of a stripe that is not imaged'))
    fig.legend(handles=handles,
               loc='lower right', bbox_to_anchor=(0.985, 0.005), frameon=False, fontsize=8.5, ncol=1)
    fig.suptitle('DESI Run 2 medium-band imaging: completeness by stripe', x=0.05, y=0.985, ha='left', fontsize=12)
    fig.savefig(a.out); print('wrote', a.out)
    if a.out.endswith('.pdf'):
        fig.savefig(a.out[:-4] + '.png', dpi=110); print('wrote', a.out[:-4] + '.png')


if __name__ == '__main__':
    main()
