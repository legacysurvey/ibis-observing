#!/usr/bin/env python3
"""Set IN_IBIS/IN_HSC for the IBIS wide tiles from the DESI Run 2 imaging plan.

A wide tile is in the imaged footprint (IN_IBIS=1) when its DECam field
(2.1 x 1.9 deg) overlaps POLU+IBIS sky or an HSC-Niji+VST rectangle, and is
a DECam program tile (IN_IBIS=1, IN_HSC=0) when its field overlaps POLU+IBIS
sky and does not lie entirely inside a Niji rectangle; IN_HSC=1 therefore
marks footprint tiles whose sky is HSC-Niji+VST's alone (footprint.py; the
stripe and region tables are copies of the ones in cfht-observing/obstatus,
derived from desi_run2_imaging_plan/sample631.tex). Deep tiles are untouched. With --ngc8-row, wide tiles from ibis-tiles-extra.ecsv whose
centers fall in the given DEC range inside NGC-8 are moved into the catalog
(PRIORITY 1, EFFTIME_GOAL recomputed with the current coefficients) so that
coverage reaches the stripe's southern bound. With --add-stripes, every
extra-file wide tile whose field overlaps the POLU+IBIS sky of the named stripes
is moved into the catalog at --add-priority, and catalog tiles that become
program tiles through those stripes get the same priority (used on 2026-11-19
to add NGC-9 and SGC-9 at PRIORITY 0.1). Edits are line-based so that
untouched rows keep their exact text. Requires numpy only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from footprint import DECAM_INSIDE, DECAM_OUTER, Footprint

HERE = Path(__file__).resolve().parent
# EFFTIME_GOAL = base * 10**(0.8 * k * EBV_MED), capped at 2000 s (ibis_exptimes.pro,
# with the extinction coefficients of the 2026-09-09 update).
EFFTIME_BASE = {'M411': 377.0, 'M438': 310.0, 'M464': 259.0, 'M490': 251.0, 'M517': 200.0}
EFFTIME_COEFF = {'M411': 3.774, 'M438': 3.576, 'M464': 3.366, 'M490': 3.141, 'M517': 2.922}
EFFTIME_CAP = 2000.0


def read_lines(path: Path):
    header, rows, prefix = None, [], []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if header is None:
                if line.startswith('#'):
                    prefix.append(line)
                    continue
                header = line.rstrip('\n').split()
                prefix.append(line)
                continue
            rows.append(line)
    return prefix, header, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=Path, default=HERE / 'ibis-tiles.ecsv')
    parser.add_argument('--extra', type=Path, default=HERE / 'ibis-tiles-extra.ecsv')
    parser.add_argument('--ngc8-row', type=float, nargs=2, metavar=('DEC_MIN', 'DEC_MAX'),
                        help='Move extra-file wide tiles inside NGC-8 with centers in this DEC range '
                             'into the catalog (e.g. -7.65 -7.05).')
    parser.add_argument('--ngc8-fill', type=float, nargs=2, metavar=('DEC_MIN', 'DEC_MAX'),
                        help='After --ngc8-row, greedily add further extra-file tiles with centers in this '
                             'DEC range until no POLU+IBIS sky in NGC-8 is left uncovered by the DECam field.')
    parser.add_argument('--add-stripes', nargs='+', metavar='STRIPE', default=[],
                        help='Stripes (POLU_IBIS=1 in desi2-stripes.ecsv) to populate from the extra file, e.g. NGC-9 SGC-9.')
    parser.add_argument('--add-priority', type=float, default=0.1, help='PRIORITY for the tiles of --add-stripes (default 0.1).')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    footprint = Footprint.from_plan(HERE)
    prefix, header, rows = read_lines(args.tiles)
    col = {name: i for i, name in enumerate(header)}
    parts = [r.split() for r in rows]
    wide = np.array([p[col['OBJECT']].startswith('IBIS_wide_') for p in parts])
    ra = np.array([float(p[col['RA']]) for p in parts]); dec = np.array([float(p[col['DEC']]) for p in parts])
    in_ibis = np.array([int(float(p[col['IN_IBIS']])) for p in parts]); in_hsc = np.array([int(float(p[col['IN_HSC']])) for p in parts])
    program = footprint.program_tiles(ra, dec, DECAM_OUTER, DECAM_INSIDE) & wide
    in_footprint = footprint.in_footprint(ra, dec, DECAM_OUTER) & wide
    new_ibis = np.where(wide, in_footprint.astype(int), in_ibis)
    new_hsc = np.where(wide, (in_footprint & ~program).astype(int), in_hsc)
    # Re-enabled tiles that were parked with PRIORITY 0 need a priority to be scheduled.
    priority = np.array([float(p[col['PRIORITY']]) for p in parts])
    revive = program & (priority == 0)
    added = None
    newprog = np.zeros(len(parts), bool)
    if args.add_stripes:
        srows = {r['STRIPE']: r for r in footprint_stripe_rows(HERE)}
        missing = [n for n in args.add_stripes if n not in srows or not int(srows[n]['POLU_IBIS'])]
        if missing:
            raise SystemExit(f'--add-stripes: not POLU_IBIS=1 stripes in desi2-stripes.ecsv: {missing}')
        added = Footprint([(float(srows[n]['RA_MIN']), float(srows[n]['RA_MAX']), float(srows[n]['DEC_MIN']), float(srows[n]['DEC_MAX']))
                           for n in args.add_stripes], [], [])
        # catalog tiles that become program tiles because of the added stripes
        newprog = program & ~((in_ibis == 1) & (in_hsc == 0)) & added.near_polu(ra, dec, DECAM_OUTER)
        revive &= ~newprog
    changed = (new_ibis != in_ibis) | (new_hsc != in_hsc) | revive | newprog
    print(f'wide tiles: {wide.sum()}; program before: {((in_ibis == 1) & (in_hsc == 0) & wide).sum()}, '
          f'after: {program.sum()}')
    print(f'  IN_IBIS 0->1: {((in_ibis == 0) & (new_ibis == 1)).sum()}, 1->0: {((in_ibis == 1) & (new_ibis == 0)).sum()}; '
          f'IN_HSC 1->0: {((in_hsc == 1) & (new_hsc == 0)).sum()}, 0->1: {((in_hsc == 0) & (new_hsc == 1)).sum()}; '
          f'PRIORITY 0->1 for program tiles: {revive.sum()}')
    out_rows = []
    for i, (line, p) in enumerate(zip(rows, parts)):
        if changed[i]:
            tokens = line.rstrip('\n').split(' ')
            # Replace tokens by column position, keeping the original spacing.
            nonblank = [k for k, tok in enumerate(tokens) if tok != '']
            tokens[nonblank[col['IN_IBIS']]] = str(new_ibis[i])
            tokens[nonblank[col['IN_HSC']]] = str(new_hsc[i])
            if revive[i]:
                tokens[nonblank[col['PRIORITY']]] = '1.0'
            if newprog[i]:
                tokens[nonblank[col['PRIORITY']]] = f'{args.add_priority:g}'
            line = ' '.join(tokens) + '\n'
        out_rows.append(line)

    moved = 0
    if args.ngc8_row:
        _, eheader, erows = read_lines(args.extra)
        ecol = {name: i for i, name in enumerate(eheader)}
        if eheader != header:
            raise SystemExit('ibis-tiles-extra.ecsv columns differ from ibis-tiles.ecsv')
        existing = {p[col['OBJECT']] for p in parts}
        ngc8 = [r for r in footprint_stripe_rows(HERE) if r['STRIPE'] == 'NGC-8'][0]
        ra0, ra1 = float(ngc8['RA_MIN']), float(ngc8['RA_MAX'])
        d0, d1 = float(ngc8['DEC_MIN']), float(ngc8['DEC_MAX'])
        eparts = [line.split() for line in erows]
        era = np.array([float(p[ecol['RA']]) for p in eparts]); edec = np.array([float(p[ecol['DEC']]) for p in eparts])
        ewide = np.array([p[ecol['OBJECT']].startswith('IBIS_wide_') and p[ecol['OBJECT']] not in existing for p in eparts])
        hra = DECAM_OUTER[0] / np.cos(np.deg2rad(edec))
        inside_ra = (((era - ra0) % 360.0) <= ((ra1 - ra0) % 360.0) + hra) | (((ra0 - era) % 360.0) <= hra)
        lo, hi = sorted(args.ngc8_row)
        select = ewide & inside_ra & (edec >= lo) & (edec <= hi)
        if args.ngc8_fill:
            # Greedy fill: cover every POLU+IBIS cell inside NGC-8 with the DECam field.
            # The tile grid is the same in every filter, so choose on one filter and
            # then take the tiles at the same positions in all filters.
            flo, fhi = sorted(args.ngc8_fill)
            efilter = np.array([p[ecol['FILTER']] for p in eparts])
            one = efilter == efilter[ewide][0]
            pfilter = np.array([p[col['FILTER']] for p in parts])
            target = footprint.polu & footprint.rectangle(ra0, ra1, d0, d1)
            covered = coverage(footprint, np.concatenate([ra[program & (pfilter == efilter[ewide][0])], era[select & one]]),
                               np.concatenate([dec[program & (pfilter == efilter[ewide][0])], edec[select & one]]))
            uncovered = target & ~covered
            candidates = np.flatnonzero(ewide & one & inside_ra & (edec >= flo) & (edec <= fhi) & ~select)
            chosen = []
            while uncovered.any() and len(candidates):
                gains = np.array([int((uncovered & coverage(footprint, [era[i]], [edec[i]])).sum()) for i in candidates])
                best = int(np.argmax(gains))
                if gains[best] == 0:
                    break
                i = candidates[best]
                chosen.append((round(era[i], 4), round(edec[i], 4)))
                uncovered &= ~coverage(footprint, [era[i]], [edec[i]])
                candidates = np.delete(candidates, best)
            positions = set(chosen)
            for i in np.flatnonzero(ewide):
                if (round(era[i], 4), round(edec[i], 4)) in positions:
                    select[i] = True
            print(f'NGC-8 fill: {len(chosen)} positions per filter; '
                  f'{footprint.area(uncovered):.2f} sq deg of NGC-8 left uncovered')
        keep_extra = []
        for i, line in enumerate(erows):
            if not select[i]:
                keep_extra.append(line); continue
            p = eparts[i]
            f = p[ecol['FILTER']]
            ebv = float(p[ecol['EBV_MED']])
            goal = min(EFFTIME_BASE[f] * 10 ** (0.8 * EFFTIME_COEFF[f] * ebv), EFFTIME_CAP)
            fields = {'PRIORITY': '1.0', 'PASS': p[ecol['PASS']], 'IN_IBIS': '1', 'IN_HSC': '0',
                      'EFFTIME_GOAL': f'{goal:.2f}'}
            new = list(p)
            for k, v in fields.items():
                new[col[k]] = v
            out_rows.append(' '.join(new) + '\n')
            moved += 1
        print(f'NGC-8 row: moved {moved} wide tiles from {args.extra.name} '
              f'({moved // 5} per filter; DEC {lo} to {hi}{" plus fill" if args.ngc8_fill else ""})')
        if not args.dry_run and moved:
            eprefix, _, _ = read_lines(args.extra)
            args.extra.write_text(''.join(eprefix) + ''.join(keep_extra), encoding='utf-8')
    if added is not None:
        print(f'{", ".join(args.add_stripes)}: {newprog.sum()} catalog tiles become program tiles at PRIORITY {args.add_priority:g}')
        eprefix, eheader, erows = read_lines(args.extra)
        if eheader != header:
            raise SystemExit('ibis-tiles-extra.ecsv columns differ from ibis-tiles.ecsv')
        existing = {p[col['OBJECT']] for p in parts}
        eparts = [line.split() for line in erows]
        era = np.array([float(p[col['RA']]) for p in eparts]); edec = np.array([float(p[col['DEC']]) for p in eparts])
        select = (np.array([p[col['OBJECT']].startswith('IBIS_wide_') and p[col['OBJECT']] not in existing for p in eparts])
                  & footprint.program_tiles(era, edec, DECAM_OUTER, DECAM_INSIDE) & added.near_polu(era, edec, DECAM_OUTER))
        keep_extra, nadd = [], 0
        for i, line in enumerate(erows):
            if not select[i]:
                keep_extra.append(line); continue
            p = list(eparts[i]); f = p[col['FILTER']]
            goal = min(EFFTIME_BASE[f] * 10 ** (0.8 * EFFTIME_COEFF[f] * float(p[col['EBV_MED']])), EFFTIME_CAP)
            for k, v in {'PRIORITY': f'{args.add_priority:g}', 'IN_IBIS': '1', 'IN_HSC': '0', 'EFFTIME_GOAL': f'{goal:.2f}'}.items():
                p[col[k]] = v
            out_rows.append(' '.join(p) + '\n'); nadd += 1
        print(f'{", ".join(args.add_stripes)}: moved {nadd} wide tiles from {args.extra.name} ({nadd // 5} per filter) at PRIORITY {args.add_priority:g}')
        if not args.dry_run and nadd:
            args.extra.write_text(''.join(eprefix) + ''.join(keep_extra), encoding='utf-8')
    if args.dry_run:
        print('dry run: nothing written')
        return
    args.tiles.write_text(''.join(prefix) + ''.join(out_rows), encoding='utf-8')
    print(f'Wrote {args.tiles} ({len(out_rows)} rows)')


def coverage(footprint: Footprint, ra, dec) -> np.ndarray:
    """Raster cells covered by DECam fields centered on the given tiles."""
    cov = np.zeros(footprint.polu.shape, dtype=bool)
    hra, hdec = DECAM_OUTER
    for r0, d0 in zip(np.asarray(ra, float), np.asarray(dec, float)):
        i0 = int(np.floor((d0 - hdec - footprint.dec0) / footprint.cell))
        i1 = int(np.ceil((d0 + hdec - footprint.dec0) / footprint.cell))
        w = hra / np.cos(np.deg2rad(d0))
        j = np.arange(int(np.floor((r0 - w) / footprint.cell)), int(np.ceil((r0 + w) / footprint.cell))) % footprint.n_ra
        cov[max(i0, 0):min(i1, footprint.n_dec), j] = True
    return cov


def footprint_stripe_rows(root: Path):
    with (root / 'desi2-stripes.ecsv').open(newline='', encoding='utf-8') as handle:
        lines = [line for line in handle if not line.startswith('#')]
    return list(csv.DictReader(lines, delimiter=' ', skipinitialspace=True))


if __name__ == '__main__':
    main()
