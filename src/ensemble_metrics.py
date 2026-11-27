"""Deterministic structural ensemble metrics and independent QC utilities.

This module extracts the core, Colab-independent post-processing and quality-control
calculations from the VAMP721–PICALM1a BioEmu-2 analysis notebook into reusable,
unit-tested Python functions:

- DSSP secondary-structure aggregation and contiguous α-helix run lengths
- Distance-based interchain contact tensors and multi-cutoff sensitivity analysis
- Independent heavy-atom steric clash detection via SciPy ``cKDTree``
- Interface buried surface area (BSA) conversion from Shrake–Rupley SASA
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = [
    "ClashRecord",
    "compute_interchain_contacts",
    "compute_interface_bsa_angstrom2",
    "compute_pairwise_distances_nm",
    "cutoff_sensitivity_analysis",
    "detect_interchain_clashes",
    "longest_true_run",
    "summarize_dssp_region",
]


@dataclass(frozen=True)
class ClashRecord:
    """Single interchain heavy-atom steric clash detected strictly below a distance cutoff."""

    frame_index: int
    atom_a: str
    atom_b: str
    distance_angstrom: float


def _validate_coord_pair(
    coords_a_nm: np.ndarray,
    coords_b_nm: np.ndarray,
    *,
    allow_empty_atoms: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired 3D coordinate tensors ``(n_frames, n_atoms, 3)`` in nanometers."""
    a = np.asarray(coords_a_nm, dtype=float)
    b = np.asarray(coords_b_nm, dtype=float)
    if a.ndim != 3 or b.ndim != 3 or a.shape[2] != 3 or b.shape[2] != 3:
        raise ValueError(
            f"Expected 3D coordinate arrays (n_frames, n_atoms, 3); got {a.shape}, {b.shape}"
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Frame count mismatch: {a.shape[0]} vs {b.shape[0]}")
    if a.shape[0] == 0:
        raise ValueError("Coordinate arrays must contain at least one frame")
    if not allow_empty_atoms and (a.shape[1] == 0 or b.shape[1] == 0):
        raise ValueError("Coordinate arrays must contain at least one atom per chain")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Coordinate arrays must contain only finite numbers")
    return a, b


def longest_true_run(mask: Sequence[bool] | np.ndarray) -> int:
    """Return the length of the longest contiguous ``True`` run in a 1D boolean sequence."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {arr.shape}")

    best = 0
    current = 0
    for item in arr:
        if item:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def summarize_dssp_region(
    dssp_codes: np.ndarray,
    residue_numbers: Sequence[int],
    start_res: int = 136,
    end_res: int = 185,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize DSSP helix occupancy across frames and residues in a target range.

    Parameters
    ----------
    dssp_codes:
        2D array of shape ``(n_frames, n_residues)`` containing single-character DSSP codes
        (e.g. ``"H"``, ``"G"``, ``"I"``, ``"E"``, ``"C"``).
    residue_numbers:
        1D sequence of 1-based canonical residue numbers aligned with axis 1 of ``dssp_codes``.
    start_res, end_res:
        Inclusive residue number boundaries of the region to evaluate.

    Returns
    -------
    per_frame_df:
        DataFrame with ``frame_index``, ``fraction_H``, and ``longest_contiguous_H``.
    per_residue_df:
        DataFrame with ``residue``, ``fraction_H``, ``fraction_G``, and ``fraction_I``.
    """
    codes = np.asarray(dssp_codes)
    if codes.ndim != 2:
        raise ValueError(f"dssp_codes must be 2D (n_frames, n_residues), got shape {codes.shape}")
    if codes.shape[0] == 0:
        raise ValueError("dssp_codes must contain at least one frame")
    if codes.shape[1] != len(residue_numbers):
        raise ValueError(
            f"residue_numbers length ({len(residue_numbers)}) must match "
            f"dssp_codes.shape[1] ({codes.shape[1]})"
        )

    res_arr = np.asarray(residue_numbers, dtype=int)
    region_mask = (res_arr >= start_res) & (res_arr <= end_res)
    if not np.any(region_mask):
        raise ValueError(f"No residues found in inclusive range [{start_res}, {end_res}]")

    region_codes = codes[:, region_mask]
    region_res = res_arr[region_mask]
    mask_h = region_codes == "H"

    per_frame_df = pd.DataFrame(
        {
            "frame_index": np.arange(codes.shape[0], dtype=int),
            "fraction_H": mask_h.mean(axis=1),
            "longest_contiguous_H": [longest_true_run(row) for row in mask_h],
        }
    )
    per_residue_df = pd.DataFrame(
        {
            "residue": region_res,
            "fraction_H": mask_h.mean(axis=0),
            "fraction_G": (region_codes == "G").mean(axis=0),
            "fraction_I": (region_codes == "I").mean(axis=0),
        }
    )
    return per_frame_df, per_residue_df


def compute_pairwise_distances_nm(
    coords_a_nm: np.ndarray,
    coords_b_nm: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean distances (nm) between two chains across frames.

    Parameters
    ----------
    coords_a_nm:
        Array of shape ``(n_frames, n_a, 3)`` in nanometers.
    coords_b_nm:
        Array of shape ``(n_frames, n_b, 3)`` in nanometers.

    Returns
    -------
    np.ndarray
        Distance tensor of shape ``(n_frames, n_a, n_b)`` in nanometers.
    """
    a, b = _validate_coord_pair(coords_a_nm, coords_b_nm)
    diff = a[:, :, None, :] - b[:, None, :, :]
    return np.linalg.norm(diff, axis=-1)


def compute_interchain_contacts(
    coords_a_nm: np.ndarray,
    coords_b_nm: np.ndarray,
    cutoff_angstrom: float = 8.0,
) -> dict[str, np.ndarray]:
    """Compute interchain contact tensors (strictly ``< cutoff_angstrom``) and per-frame counts."""
    if cutoff_angstrom <= 0:
        raise ValueError(f"cutoff_angstrom must be positive, got {cutoff_angstrom}")

    dist_nm = compute_pairwise_distances_nm(coords_a_nm, coords_b_nm)
    contact_map = dist_nm < (cutoff_angstrom / 10.0)
    return {
        "contact_map": contact_map,
        "contact_frequency_2d": contact_map.mean(axis=0),
        "n_pairs_per_frame": contact_map.sum(axis=(1, 2)),
        "n_a_residues_per_frame": contact_map.any(axis=2).sum(axis=1),
        "n_b_residues_per_frame": contact_map.any(axis=1).sum(axis=1),
        "a_contact_frequency": contact_map.any(axis=2).mean(axis=0),
        "b_contact_frequency": contact_map.any(axis=1).mean(axis=0),
    }


def cutoff_sensitivity_analysis(
    coords_a_nm: np.ndarray,
    coords_b_nm: np.ndarray,
    cutoffs_angstrom: Iterable[float] = (6.0, 8.0, 10.0),
) -> pd.DataFrame:
    """Evaluate interchain contact pair counts across multiple strict ``<`` distance cutoffs."""
    cutoffs = list(cutoffs_angstrom)
    if not cutoffs or any(c <= 0 for c in cutoffs):
        raise ValueError("All distance cutoffs must be positive")

    dist_nm = compute_pairwise_distances_nm(coords_a_nm, coords_b_nm)
    data: dict[str, np.ndarray] = {"frame_index": np.arange(dist_nm.shape[0], dtype=int)}
    for cutoff in cutoffs:
        label = f"pairs_lt{int(cutoff)}A" if float(cutoff).is_integer() else f"pairs_lt{cutoff:g}A"
        if label in data:
            raise ValueError(f"Duplicate cutoff specified: {cutoff}")
        data[label] = (dist_nm < (cutoff / 10.0)).sum(axis=(1, 2))
    return pd.DataFrame(data)


def detect_interchain_clashes(
    coords_a_nm: np.ndarray,
    coords_b_nm: np.ndarray,
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    max_distance_angstrom: float = 2.2,
) -> list[ClashRecord]:
    """Detect interchain heavy-atom pairs strictly closer than ``max_distance_angstrom``.

    This mirrors the independent post-hoc steric QC that flagged residual clashes
    (``< 2.2 Å``) even in frames passing model-level ``check_physicality``.
    """
    a, b = _validate_coord_pair(coords_a_nm, coords_b_nm, allow_empty_atoms=True)
    if a.shape[1] != len(labels_a) or b.shape[1] != len(labels_b):
        raise ValueError("Atom label lengths must match coordinate arrays")
    if max_distance_angstrom <= 0:
        raise ValueError(f"max_distance_angstrom must be positive, got {max_distance_angstrom}")
    if a.shape[1] == 0 or b.shape[1] == 0:
        return []

    clashes: list[ClashRecord] = []
    for frame in range(a.shape[0]):
        tree_a = cKDTree(a[frame] * 10.0)
        tree_b = cKDTree(b[frame] * 10.0)
        near = tree_a.sparse_distance_matrix(
            tree_b,
            max_distance=max_distance_angstrom,
            output_type="dict",
        )
        # sparse_distance_matrix is inclusive (<= max_distance); enforce strict (<)
        strict_pairs = [
            ((i, j), float(dist_a))
            for (i, j), dist_a in near.items()
            if dist_a < max_distance_angstrom
        ]
        for (i, j), dist_a in sorted(strict_pairs, key=lambda kv: (kv[1], kv[0][0], kv[0][1])):
            clashes.append(
                ClashRecord(
                    frame_index=frame,
                    atom_a=str(labels_a[i]),
                    atom_b=str(labels_b[j]),
                    distance_angstrom=dist_a,
                )
            )
    return clashes


def compute_interface_bsa_angstrom2(
    sasa_complex_nm2: Sequence[float] | np.ndarray,
    sasa_a_nm2: Sequence[float] | np.ndarray,
    sasa_b_nm2: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute one-sided buried surface area (Å²) from Shrake–Rupley SASA values (nm²).

    Formula: ``BSA = (SASA_A + SASA_B - SASA_complex) * 100 / 2``.
    """
    sc = np.asarray(sasa_complex_nm2, dtype=float)
    sa = np.asarray(sasa_a_nm2, dtype=float)
    sb = np.asarray(sasa_b_nm2, dtype=float)
    if not (sc.shape == sa.shape == sb.shape):
        raise ValueError(f"SASA shapes must match: {sc.shape}, {sa.shape}, {sb.shape}")
    if not (np.isfinite(sc).all() and np.isfinite(sa).all() and np.isfinite(sb).all()):
        raise ValueError("SASA values must contain only finite numbers")
    return (sa + sb - sc) * 100.0 / 2.0
