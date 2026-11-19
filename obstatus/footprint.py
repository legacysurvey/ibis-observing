"""Which sky is CFHT (POLU) + DECam (IBIS) sky, from the DESI Run 2 imaging plan.

The plan (desi_run2_imaging_plan/sample631.tex) defines declination stripes
(obstatus/desi2-stripes.ecsv) and, inside some of them, HSC-Niji+VST
rectangles plus two dropped slivers (obstatus/footprint-regions.ecsv).
POLU+IBIS sky is the union of the stripes with POLU_IBIS=1 minus those
regions. Stripes with POLU_IBIS=0 (SGC-1) contribute nothing; NGC-9 and SGC-9,
formerly proposed extensions, are baseline POLU+IBIS stripes since 2026-11-19.

Membership is decided on a raster (default 0.05-degree cells) so that one
rule serves any camera: a tile is in the imaged footprint (IN_IBIS=1) when
its center lies within ``outer`` degrees of POLU+IBIS sky or of an
HSC-Niji+VST rectangle, and it is a program tile when its center lies within
``outer`` degrees of POLU+IBIS sky unless a box of half-size ``inside``
centered on it lies entirely inside the HSC-Niji+VST rectangles (so nothing
is spent on sky Niji covers). With ``outer`` and ``inside`` both set to the
camera's half-field, as for MegaCam and DECam, this is exactly "the field
overlaps POLU+IBIS sky and is not Niji-only". Both are (RA, DEC) half-sizes in degrees; RA extents are
scaled by 1/cos(DEC). Requires numpy only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent  # obstatus/: holds desi2-stripes.ecsv and footprint-regions.ecsv
CELL = 0.05
DEC_RANGE = (-30.0, 30.0)


def read_ecsv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        lines = [line for line in handle if not line.startswith('#')]
    return list(csv.DictReader(lines, delimiter=' ', skipinitialspace=True))


def plan_rectangles(root: Path = ROOT):
    """Return (polu_stripes, excluded) rectangle lists as (ra0, ra1, dec0, dec1)."""
    polu = []
    for row in read_ecsv_rows(root / "desi2-stripes.ecsv"):
        if int(row.get('POLU_IBIS', 1)):
            polu.append((float(row['RA_MIN']), float(row['RA_MAX']),
                         float(row['DEC_MIN']), float(row['DEC_MAX'])))
    excluded, niji = [], []
    for row in read_ecsv_rows(root / "footprint-regions.ecsv"):
        rect = (float(row['RA_MIN']), float(row['RA_MAX']), float(row['DEC_MIN']), float(row['DEC_MAX']))
        excluded.append(rect)
        if row['KIND'] == 'niji':
            niji.append(rect)
    return polu, excluded, niji


class Footprint:
    """Raster of POLU+IBIS sky and of the HSC-Niji+VST rectangles."""

    def __init__(self, polu_rects, excluded_rects, niji_rects, cell=CELL, dec_range=DEC_RANGE):
        self.cell = float(cell)
        self.dec0 = float(dec_range[0])
        self.n_ra = int(round(360.0 / self.cell))
        self.n_dec = int(round((dec_range[1] - dec_range[0]) / self.cell))
        self.dec_centers = self.dec0 + (np.arange(self.n_dec) + 0.5) * self.cell
        self.ra_centers = (np.arange(self.n_ra) + 0.5) * self.cell
        self.polu = np.zeros((self.n_dec, self.n_ra), dtype=bool)
        for rect in polu_rects:
            self.polu |= self.rectangle(*rect)
        for rect in excluded_rects:
            self.polu &= ~self.rectangle(*rect)
        self.niji = np.zeros((self.n_dec, self.n_ra), dtype=bool)
        for rect in niji_rects:
            self.niji |= self.rectangle(*rect)
        # The imaged DESI Run 2 footprint: POLU+IBIS sky plus the HSC-Niji+VST rectangles.
        self.imaged = self.polu | self.niji
        self._cache = {}

    @classmethod
    def from_plan(cls, root: Path = ROOT, **kwargs) -> 'Footprint':
        return cls(*plan_rectangles(root), **kwargs)

    def rectangle(self, ra0, ra1, dec0, dec1) -> np.ndarray:
        """Cells whose centers fall inside the rectangle; RA wraps through 360."""
        rows = (self.dec_centers > dec0) & (self.dec_centers < dec1)
        width = (ra1 - ra0) % 360.0 or 360.0
        cols = ((self.ra_centers - ra0) % 360.0) < width
        mask = np.zeros((self.n_dec, self.n_ra), dtype=bool)
        mask[np.ix_(rows, cols)] = True
        return mask

    def dilate(self, mask: np.ndarray, half_ra: float, half_dec: float) -> np.ndarray:
        """Cells within a (half_ra / cos DEC, half_dec) box of a True cell.

        One extra cell is added in each direction so that a center exactly
        ``half`` away is included despite the raster granularity.
        """
        k_dec = int(np.ceil(half_dec / self.cell)) + 1
        out = mask.copy()
        for shift in range(1, k_dec + 1):
            out[shift:] |= mask[:-shift]
            out[:-shift] |= mask[shift:]
        widths = half_ra / np.maximum(np.cos(np.deg2rad(self.dec_centers)), 1e-6) / self.cell + 1
        k_ra = int(np.ceil(widths.max()))
        source = out.copy()
        for shift in range(1, k_ra + 1):
            rows = (widths >= shift)[:, None]
            out |= rows & np.roll(source, shift, axis=1)
            out |= rows & np.roll(source, -shift, axis=1)
        return out

    def erode(self, mask: np.ndarray, half_ra: float, half_dec: float) -> np.ndarray:
        """Cells whose whole box lies inside the mask."""
        return ~self.dilate(~mask, half_ra, half_dec)

    def cell_index(self, ra, dec):
        ra = np.asarray(ra, dtype=float) % 360.0
        dec = np.asarray(dec, dtype=float)
        col = np.floor(ra / self.cell).astype(int) % self.n_ra
        row = np.clip(np.floor((dec - self.dec0) / self.cell).astype(int), 0, self.n_dec - 1)
        outside = (dec < self.dec0) | (dec >= self.dec0 + self.n_dec * self.cell)
        return row, col, outside

    def near_polu(self, ra, dec, outer) -> np.ndarray:
        """Tile centers within ``outer`` (half_ra, half_dec) of POLU+IBIS sky."""
        key = ('near', tuple(outer))
        if key not in self._cache:
            self._cache[key] = self.dilate(self.polu, *outer)
        row, col, outside = self.cell_index(ra, dec)
        return self._cache[key][row, col] & ~outside

    def in_footprint(self, ra, dec, outer) -> np.ndarray:
        """Tile centers within ``outer`` (half_ra, half_dec) of the imaged footprint."""
        key = ('footprint', tuple(outer))
        if key not in self._cache:
            self._cache[key] = self.dilate(self.imaged, *outer)
        row, col, outside = self.cell_index(ra, dec)
        return self._cache[key][row, col] & ~outside

    def inside_niji(self, ra, dec, inside) -> np.ndarray:
        """Tiles whose ``inside`` (half_ra, half_dec) box lies entirely in Niji rectangles."""
        key = ('inside', tuple(inside))
        if key not in self._cache:
            self._cache[key] = self.erode(self.niji, *inside)
        row, col, outside = self.cell_index(ra, dec)
        return self._cache[key][row, col] & ~outside

    def program_tiles(self, ra, dec, outer, inside) -> np.ndarray:
        """Tiles CFHT/DECam should observe: near POLU+IBIS sky and not Niji-only."""
        return self.near_polu(ra, dec, outer) & ~self.inside_niji(ra, dec, inside)

    def area(self, mask: np.ndarray) -> float:
        """Solid angle of a raster mask in square degrees."""
        cell_area = np.cos(np.deg2rad(self.dec_centers)) * self.cell ** 2
        return float((mask.sum(axis=1) * cell_area).sum())


# Camera footprints as (half_ra, half_dec) in degrees.
# For both cameras the outer margin equals the half-field, so the rule is simply
# "the field overlaps POLU+IBIS sky and does not lie entirely inside a Niji rectangle".
MEGACAM_OUTER = (0.5, 0.5)      # the ~1 x 1 deg MegaCam field
MEGACAM_INSIDE = (0.5, 0.5)
DECAM_OUTER = (1.05, 0.95)      # the ~2.1 x 1.9 deg DECam field
DECAM_INSIDE = (1.05, 0.95)
