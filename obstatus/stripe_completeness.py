#!/usr/bin/env python
"""Completeness of each DESI Run 2 imaging stripe in each medium-band filter.

Counts tiles, not sky area: for every stripe and filter, the number of program
tiles marked DONE in the tile files over the number of program tiles.

  DECam (IBIS): ibis-tiles.ecsv, wide tiles (OBJECT IBIS_wide_*) with IN_IBIS=1, IN_HSC=0
  CFHT  (POLU): cfht-tiles.ecsv, tiles with IN_IBIS=1, IN_HSC=0
  HSC   (Niji): optional tiling table (--hsc-tiles) with ra, dec and, once it
                exists, a done column; VST can be added the same way (--vst-tiles)

Each tile is assigned to exactly one stripe: the stripe in its Galactic cap
(NGC: 90 < RA < 270) whose central declination is nearest to the tile centre,
among the stripes that camera images (POLU_IBIS=1 in desi2-stripes.ecsv for
CFHT and DECam; the stripes with an HSC-Niji+VST rectangle in
footprint-regions.ecsv for HSC and VST).  Tiles that spill over a stripe edge
are therefore counted once, in the stripe holding their centre.

The possible extensions NGC-9 and SGC-9 are not in the program; for them every
tile whose field overlaps the stripe is counted as planned (DECam: from
ibis-tiles-extra.ecsv too).  --no-extended leaves them out.

  python stripe_completeness.py
  python stripe_completeness.py --cfht-tiles ~/cfht-observing/obstatus/cfht-tiles.ecsv --markdown
"""
import argparse, os, sys
import numpy as np
from astropy.table import Table, vstack

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from footprint import Footprint, MEGACAM_OUTER, DECAM_OUTER
DEFAULT_CFHT = os.path.join(HERE, '..', '..', 'cfht-observing', 'obstatus', 'cfht-tiles.ecsv')


def is_ngc(ra):
    ra = np.asarray(ra, float) % 360
    return (ra > 90) & (ra < 270)


def assign_stripes(ra, dec, names, centers):
    """Name of the stripe (from names/centers) nearest in Dec within each tile's cap."""
    names = np.asarray(names); centers = np.asarray(centers, float)
    ngc_stripe = np.array([n.startswith('NGC') for n in names])
    ngc_tile = is_ngc(ra); dec = np.asarray(dec, float)
    out = np.full(len(dec), '', dtype='U8')
    for cap in (True, False):
        s = ngc_stripe == cap; t = ngc_tile == cap
        if s.any() and t.any():
            out[t] = names[s][np.argmin(np.abs(dec[t][:, None] - centers[s][None, :]), axis=1)]
    return out


def count(stripe, done, sel):
    """{stripe: (ndone, ntotal)} for the selected tiles."""
    res = {}
    for s in np.unique(stripe[sel]):
        m = sel & (stripe == s)
        res[s] = (int(done[m].sum()), int(m.sum()))
    return res


def read_generic(path):
    """ra, dec, done (zeros if there is no done column) from a csv/ecsv/fits tiling table."""
    t = Table.read(path, comment='#') if path.endswith('.csv') else Table.read(path)
    low = {c.lower(): c for c in t.colnames}
    done = np.asarray(t[low['done']]) > 0 if 'done' in low else np.zeros(len(t), bool)
    return np.asarray(t[low['ra']], float), np.asarray(t[low['dec']], float), done, 'done' in low


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ibis-tiles', default=os.path.join(HERE, 'ibis-tiles.ecsv'))
    p.add_argument('--cfht-tiles', default=DEFAULT_CFHT)
    p.add_argument('--hsc-tiles', help='HSC-Niji tiling table (ra, dec[, done])')
    p.add_argument('--vst-tiles', help='VST tiling table (ra, dec[, done])')
    p.add_argument('--stripes', default=os.path.join(HERE, 'desi2-stripes.ecsv'))
    p.add_argument('--regions', default=os.path.join(HERE, 'footprint-regions.ecsv'))
    p.add_argument('--ibis-extra', default=os.path.join(HERE, 'ibis-tiles-extra.ecsv'))
    p.add_argument('--no-extended', action='store_true', help='leave out the NGC-9 and SGC-9 extensions')
    p.add_argument('--markdown', action='store_true', help='print a Markdown table')
    p.add_argument('--out', help='also write the table to this file')
    a = p.parse_args()

    st = Table.read(a.stripes); rg = Table.read(a.regions)
    ext = np.array([str(n) == 'extended' for n in st['NOTE']]) & (not a.no_extended)
    polu = st[(np.asarray(st['POLU_IBIS']) == 1) | ext]
    extfp = Footprint([(s['RA_MIN'], s['RA_MAX'], s['DEC_MIN'], s['DEC_MAX']) for s in st[ext]], [], [])

    def in_ext(t, half):         # tiles whose field overlaps an extension stripe
        if not ext.any(): return np.zeros(len(t), bool)
        return extfp.near_polu(np.asarray(t['RA'], float), np.asarray(t['DEC'], float), half)
    niji_names = [n for n in st['STRIPE'] if n in set(rg['NAME'][rg['KIND'] == 'niji'])]
    niji = st[[n in niji_names for n in st['STRIPE']]]

    cols = []          # (header, {stripe: (done, total)}, has status)
    if os.path.exists(a.cfht_tiles):
        t = Table.read(a.cfht_tiles)
        prog = ((np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)) | in_ext(t, MEGACAM_OUTER)
        stripe = assign_stripes(t['RA'], t['DEC'], polu['STRIPE'], polu['DEC_CENTER'])
        done = np.asarray(t['DONE']) > 0
        for f in sorted(set(t['FILTER'])):
            cols.append(('CFHT ' + f, count(stripe, done, prog & (np.asarray(t['FILTER']) == f)), True))
    else:
        print('No CFHT tile file at %s (use --cfht-tiles)' % a.cfht_tiles, file=sys.stderr)
    t = Table.read(a.ibis_tiles)
    if ext.any() and os.path.exists(a.ibis_extra):
        e = Table.read(a.ibis_extra)
        t = vstack([t, e[in_ext(e, DECAM_OUTER)][t.colnames]], metadata_conflicts='silent')
    wide = np.array([o.startswith('IBIS_wide') for o in t['OBJECT']])
    prog = wide & (((np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)) | in_ext(t, DECAM_OUTER))
    stripe = assign_stripes(t['RA'], t['DEC'], polu['STRIPE'], polu['DEC_CENTER'])
    done = np.asarray(t['DONE']) > 0
    for f in sorted(set(t['FILTER'])):
        cols.append(('DECam ' + f, count(stripe, done, prog & (np.asarray(t['FILTER']) == f)), True))
    for label, path in (('HSC', a.hsc_tiles), ('VST', a.vst_tiles)):
        if path:
            ra, dec, done, has = read_generic(path)
            stripe = assign_stripes(ra, dec, niji['STRIPE'], niji['DEC_CENTER'])
            cols.append((label, count(stripe, done, np.ones(len(ra), bool)), has))

    def cell(c, s):
        if s not in c[1]: return '-'
        d, n = c[1][s]
        return '%d/%d %5.1f%%' % (d, n, 100.0 * d / n) if c[2] else '0/%d (no status)' % n
    rows = [[s] + [cell(c, s) for c in cols] for s in st['STRIPE'] if any(s in c[1] for c in cols)]
    tot = ['Total']
    for c in cols:
        d = sum(v[0] for v in c[1].values()); n = sum(v[1] for v in c[1].values())
        tot.append(('%d/%d %5.1f%%' % (d, n, 100.0 * d / n)) if c[2] else '0/%d (no status)' % n)
    rows.append(tot)
    head = ['Stripe'] + [c[0] for c in cols]
    w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    if a.markdown:
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|']
        lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    else:
        lines = ['  '.join(h.ljust(x) if i == 0 else h.rjust(x) for i, (h, x) in enumerate(zip(r, w))) for r in [head] + rows]
    text = '\n'.join(lines)
    print(text)
    if a.out:
        open(a.out, 'w').write(text + '\n')


if __name__ == '__main__':
    main()
