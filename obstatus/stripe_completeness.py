#!/usr/bin/env python
"""Completeness of each DESI Run 2 imaging stripe in each medium-band filter.

Counts tiles, not sky area: for every stripe and filter, the number of program
tiles marked DONE in the tile files over the number of program tiles.

  DECam (IBIS): ibis-tiles.ecsv, wide tiles (OBJECT IBIS_wide_*) with IN_IBIS=1, IN_HSC=0
  CFHT  (POLU): cfht-tiles.ecsv, tiles with IN_IBIS=1, IN_HSC=0
  HSC   (Niji): survey_status.csv of the Niji wide-observing-strategy repository
                (--niji-repo): one row per shot, done = USABLE_LOOSE (seeing <= 2.0
                arcsec; USABLE, seeing <= 1.3 arcsec, with --niji-strict).  Each shot is one exposure through the MBQ1 quadrant
                filter, i.e. about a quarter of the field in each of 413/439/465/490,
                so one column serves the four filters.  Only shots centred within
                0.75 deg of an HSC-Niji+VST rectangle of footprint-regions.ecsv count.
                VST (the fifth band there) can be added with --vst-tiles once a
                table with ra, dec and done exists.

Each tile is assigned to exactly one stripe.  A CFHT or DECam tile goes to the
stripe whose CFHT+DECam sky (within the stripe's unique 2.8-deg band) its field
overlaps, or failing that is nearest; if two qualify, to the one nearer in
declination.  An edge tile of NGC-7 whose centre lies in the band of NGC-6 (all
HSC-Niji+VST) is therefore counted in NGC-7, and NGC-6 has no CFHT or DECam tiles.
HSC shots go to the nearest stripe that has an HSC-Niji+VST rectangle.

  python stripe_completeness.py
  python stripe_completeness.py --cfht-tiles ~/cfht-observing/obstatus/cfht-tiles.ecsv --markdown
"""
import argparse, csv, os, sys
import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from footprint import Footprint, plan_rectangles, MEGACAM_OUTER, DECAM_OUTER
DEFAULT_NIJI = os.path.join(os.path.expanduser('~'), 'Imaging', 'wide-observing-strategy')
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


def assign_by_overlap(ra, dec, st, half, root):
    """Stripe whose CFHT+DECam sky, within its unique 2.8-deg band, is nearest the field.

    Fields (half-sizes ``half``) overlapping the CFHT+DECam sky of a stripe's unique band go to
    that stripe (nearest in Dec if two); the remaining edge tiles go to the first stripe reached
    as the field is grown in steps of 0.25 deg.
    """
    from pathlib import Path
    _, excluded, _ = plan_rectangles(Path(root))
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    polu = st[np.asarray(st['POLU_IBIS']) == 1]
    fps = [Footprint([(float(s['RA_MIN']), float(s['RA_MAX']), float(s['DEC_CENTER']) - 1.4, float(s['DEC_CENTER']) + 1.4)], excluded, [])
           for s in polu]
    out = assign_stripes(ra, dec, polu['STRIPE'], polu['DEC_CENTER']); todo = np.ones(len(ra), bool)
    for grow in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        if not todo.any(): break
        idx = np.flatnonzero(todo); best = np.full(len(idx), np.inf); name = out[idx].copy()
        for s, fp in zip(polu, fps):
            hit = fp.near_polu(ra[idx], dec[idx], (half[0] + grow, half[1] + grow)); d = np.abs(dec[idx] - float(s['DEC_CENTER']))
            take = hit & (d < best)
            name[take] = s['STRIPE']; best[take] = d[take]
        got = np.isfinite(best)
        out[idx[got]] = name[got]; todo[idx[got]] = False
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
    p.add_argument('--niji-repo', default=DEFAULT_NIJI, help='clone of hsc-niji-survey/wide-observing-strategy')
    p.add_argument('--niji-strict', action='store_true', help='count only HSC shots with seeing <= 1.3 arcsec (default: the loose 2.0 arcsec limit)')
    p.add_argument('--hsc-tiles', help='HSC-Niji tiling table (ra, dec[, done])')
    p.add_argument('--vst-tiles', help='VST tiling table (ra, dec[, done])')
    p.add_argument('--stripes', default=os.path.join(HERE, 'desi2-stripes.ecsv'))
    p.add_argument('--regions', default=os.path.join(HERE, 'footprint-regions.ecsv'))
    p.add_argument('--markdown', action='store_true', help='print a Markdown table')
    p.add_argument('--out', help='also write the table to this file')
    a = p.parse_args()

    st = Table.read(a.stripes); rg = Table.read(a.regions)
    polu = st[st['POLU_IBIS'] == 1]
    niji_names = [n for n in st['STRIPE'] if n in set(rg['NAME'][rg['KIND'] == 'niji'])]
    niji = st[[n in niji_names for n in st['STRIPE']]]

    cols = []          # (header, {stripe: (done, total)}, has status)
    if os.path.exists(a.cfht_tiles):
        t = Table.read(a.cfht_tiles)
        prog = (np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)
        stripe = assign_by_overlap(t['RA'], t['DEC'], st, MEGACAM_OUTER, os.path.dirname(a.stripes))
        done = np.asarray(t['DONE']) > 0
        for f in sorted(set(t['FILTER'])):
            cols.append(('CFHT ' + f, count(stripe, done, prog & (np.asarray(t['FILTER']) == f)), True))
    else:
        print('No CFHT tile file at %s (use --cfht-tiles)' % a.cfht_tiles, file=sys.stderr)
    t = Table.read(a.ibis_tiles)
    wide = np.array([o.startswith('IBIS_wide') for o in t['OBJECT']])
    prog = wide & (np.asarray(t['IN_IBIS']) == 1) & (np.asarray(t['IN_HSC']) == 0)
    stripe = assign_by_overlap(t['RA'], t['DEC'], st, DECAM_OUTER, os.path.dirname(a.stripes))
    done = np.asarray(t['DONE']) > 0
    for f in sorted(set(t['FILTER'])):
        cols.append(('DECam ' + f, count(stripe, done, prog & (np.asarray(t['FILTER']) == f)), True))
    status = os.path.join(a.niji_repo, 'survey_status.csv')
    if os.path.exists(status) and not a.hsc_tiles:
        rows = list(csv.DictReader(open(status)))
        ra = np.array([float(r['RA']) for r in rows]); dec = np.array([float(r['DEC']) for r in rows])
        done = np.array([r['USABLE' if a.niji_strict else 'USABLE_LOOSE'] == '1' for r in rows])
        near = np.zeros(len(ra), bool)
        for r in rg[rg['KIND'] == 'niji']:
            pad = 0.75 / np.cos(np.radians(dec)); w = (r['RA_MAX'] - r['RA_MIN']) % 360
            near |= (dec > r['DEC_MIN'] - 0.75) & (dec < r['DEC_MAX'] + 0.75) & (((ra - r['RA_MIN'] + pad) % 360) < w + 2 * pad)
        stripe = assign_stripes(ra, dec, niji['STRIPE'], niji['DEC_CENTER'])
        cols.append(('HSC MBQ1 shots', count(stripe, done, near), True))
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
