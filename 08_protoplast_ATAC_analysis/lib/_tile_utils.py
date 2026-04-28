"""
Shared tile-level utilities for v4 pipeline (03b–03e).

Provides native-ACR tile masking: restricts 180-tile arrays to tiles
whose centers fall within the original (pre-resized) ACR boundaries.
"""

import numpy as np
import pandas as pd

TILE_SIZE = 10
CONTEXT_RADIUS = 100
N_TILES = 180
# Tile center positions within the 2000bp resized window (bp 105, 115, ..., 1895)
TILE_BP = np.arange(N_TILES) * TILE_SIZE + CONTEXT_RADIUS + TILE_SIZE // 2


def build_native_tile_mask(region_strs, metadata_path, mapping_path):
    """Build a boolean mask (n_regions, 180): True if tile center is inside native ACR.

    Parameters
    ----------
    region_strs : array-like of str
        Resized coordinate strings, e.g. "chr1:2054-4054".
    metadata_path : str
        Path to ACR metadata TSV (needs chr, start, end columns — native coords).
    mapping_path : str
        Path to native-to-resized coordinate mapping TSV.

    Returns
    -------
    mask : ndarray (n_regions, 180), dtype bool
        True where tile center falls within [native_start, native_end).
    n_masked : int
        Total tiles masked out (outside native ACR).
    """
    # Load native → resized mapping
    mapping = pd.read_csv(mapping_path, sep="\t")
    nat_to_resized = dict(zip(mapping["native_str"], mapping["resized_str"]))
    resized_to_native = {v: k for k, v in nat_to_resized.items()}

    # Load native ACR coordinates
    meta = pd.read_csv(metadata_path, sep="\t", usecols=["chr", "start", "end"])
    meta["native_str"] = (meta["chr"].str.lower() + ":" +
                          meta["start"].astype(str) + "-" +
                          meta["end"].astype(str))
    native_coords = {}
    for _, row in meta.iterrows():
        native_coords[row["native_str"]] = (int(row["start"]), int(row["end"]))

    n_regions = len(region_strs)
    mask = np.ones((n_regions, N_TILES), dtype=bool)  # default: all True

    n_masked_out = 0
    for i, resized_str in enumerate(region_strs):
        # Parse resized window start
        parts = resized_str.replace(":", "-").split("-")
        resized_start = int(parts[1])

        # Get native ACR boundaries
        native_str = resized_to_native.get(resized_str)
        if native_str is None or native_str not in native_coords:
            # No mapping → keep all tiles (conservative)
            continue

        native_start, native_end = native_coords[native_str]

        # Tile centers in genomic coordinates
        tile_genomic = TILE_BP + resized_start

        # Mask: True only if tile center is within native ACR
        inside = (tile_genomic >= native_start) & (tile_genomic < native_end)
        mask[i] = inside
        n_masked_out += int((~inside).sum())

    n_total = n_regions * N_TILES
    pct_kept = 100 * (n_total - n_masked_out) / n_total
    print(f"  [NATIVE MASK] {n_total - n_masked_out:,}/{n_total:,} tiles kept "
          f"({pct_kept:.1f}%), {n_masked_out:,} flanking tiles masked")

    return mask, n_masked_out
