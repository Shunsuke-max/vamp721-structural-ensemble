"""Unit tests for deterministic structural ensemble metrics and QC utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble_metrics import (
    compute_interchain_contacts,
    compute_interface_bsa_angstrom2,
    compute_pairwise_distances_nm,
    cutoff_sensitivity_analysis,
    detect_interchain_clashes,
    longest_true_run,
    summarize_dssp_region,
)


class TestLongestTrueRun:
    def test_handles_empty_and_mixed_runs(self) -> None:
        assert longest_true_run([]) == 0
        assert longest_true_run([False, False]) == 0
        assert longest_true_run([True, True, True]) == 3
        assert longest_true_run([True, False, True, True, False, True]) == 2
        assert longest_true_run(np.array([False, True, True, False], dtype=bool)) == 2

    def test_raises_on_non_1d_mask(self) -> None:
        with pytest.raises(ValueError, match="must be 1D"):
            longest_true_run(np.array([[True, False]]))


class TestDsspSummary:
    def test_region_fractions_and_contiguous_lengths(self) -> None:
        residues = [135, 136, 137, 138, 139, 140]
        dssp = np.array(
            [
                ["C", "H", "H", "H", "C", "H"],
                ["H", "H", "G", "H", "I", "C"],
            ]
        )
        per_frame, per_res = summarize_dssp_region(dssp, residues, start_res=136, end_res=140)

        assert per_frame["fraction_H"].tolist() == pytest.approx([0.8, 0.4])
        assert per_frame["longest_contiguous_H"].tolist() == [3, 1]
        assert per_res["residue"].tolist() == [136, 137, 138, 139, 140]
        assert per_res["fraction_H"].tolist() == pytest.approx([1.0, 0.5, 1.0, 0.0, 0.5])
        assert per_res["fraction_G"].tolist() == pytest.approx([0.0, 0.5, 0.0, 0.0, 0.0])
        assert per_res["fraction_I"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.0])

    def test_raises_on_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="No residues found"):
            summarize_dssp_region(np.array([["H", "C"]]), [1, 2], start_res=10, end_res=20)
        with pytest.raises(ValueError, match="must be 2D"):
            summarize_dssp_region(np.array(["H", "C"]), [1, 2], start_res=1, end_res=2)
        with pytest.raises(ValueError, match="at least one frame"):
            summarize_dssp_region(np.empty((0, 2), dtype="<U1"), [1, 2], start_res=1, end_res=2)
        with pytest.raises(ValueError, match="residue_numbers length"):
            summarize_dssp_region(np.array([["H", "C"]]), [1, 2, 3], start_res=1, end_res=2)


class TestContactsAndSensitivity:
    def test_interchain_contacts_and_monotonic_sensitivity(self) -> None:
        # 2 frames, 2 atoms in A, 2 atoms in B (units: nm; 1 nm = 10 Å)
        coords_a = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        coords_b = np.array(
            [
                [[0.5, 0.0, 0.0], [0.9, 0.0, 0.0]],  # 5 Å, 9 Å from A0; 5 Å, 1 Å from A1
                [[0.7, 0.0, 0.0], [3.0, 0.0, 0.0]],  # 7 Å from A0; 10 Å from A1 (excluded at <10 Å)
            ]
        )

        res = compute_interchain_contacts(coords_a, coords_b, cutoff_angstrom=8.0)
        assert res["n_pairs_per_frame"].tolist() == [3, 1]
        assert res["n_a_residues_per_frame"].tolist() == [2, 1]
        assert res["n_b_residues_per_frame"].tolist() == [2, 1]
        assert res["a_contact_frequency"].tolist() == pytest.approx([1.0, 0.5])
        assert res["b_contact_frequency"].tolist() == pytest.approx([1.0, 0.5])

        sens = cutoff_sensitivity_analysis(coords_a, coords_b, cutoffs_angstrom=(6.0, 8.0, 10.0))
        assert sens["pairs_lt6A"].tolist() == [3, 0]
        assert sens["pairs_lt8A"].tolist() == [3, 1]
        assert sens["pairs_lt10A"].tolist() == [4, 1]
        # Monotonicity invariant: pairs(6 Å) <= pairs(8 Å) <= pairs(10 Å)
        assert (sens["pairs_lt6A"] <= sens["pairs_lt8A"]).all()
        assert (sens["pairs_lt8A"] <= sens["pairs_lt10A"]).all()

    def test_strict_cutoff_boundary_and_float_column_labels(self) -> None:
        coords_a = np.array([[[0.0, 0.0, 0.0]]])
        coords_b = np.array([[[0.8, 0.0, 0.0]]])  # Exactly 8.0 Å

        strict_8 = compute_interchain_contacts(coords_a, coords_b, cutoff_angstrom=8.0)
        assert strict_8["n_pairs_per_frame"].tolist() == [0]

        sens = cutoff_sensitivity_analysis(coords_a, coords_b, cutoffs_angstrom=(7.5, 8.0, 8.5))
        assert list(sens.columns) == ["frame_index", "pairs_lt7.5A", "pairs_lt8A", "pairs_lt8.5A"]
        assert sens.loc[0, ["pairs_lt7.5A", "pairs_lt8A", "pairs_lt8.5A"]].tolist() == [0, 0, 1]

    def test_validation_errors(self) -> None:
        valid = np.zeros((1, 1, 3))
        with pytest.raises(ValueError, match="must be positive"):
            compute_interchain_contacts(valid, valid, cutoff_angstrom=0.0)
        with pytest.raises(ValueError, match="Duplicate cutoff"):
            cutoff_sensitivity_analysis(valid, valid, cutoffs_angstrom=(8.0, 8.0))
        with pytest.raises(ValueError, match="Frame count mismatch"):
            compute_pairwise_distances_nm(np.zeros((1, 1, 3)), np.zeros((2, 1, 3)))
        with pytest.raises(ValueError, match="only finite numbers"):
            compute_pairwise_distances_nm(np.array([[[np.nan, 0.0, 0.0]]]), valid)


class TestStericClashDetection:
    def test_detects_strictly_sub_2_2_angstrom_clashes_only(self) -> None:
        # Frame 0: 2.02 Å clash, exact 2.20 Å boundary (excluded), 3.50 Å safe pair
        # Frame 1: 1.62 Å severe clash and 0.00 Å coincident atoms (both included, sorted by dist)
        coords_a = np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ]
        )
        coords_b = np.array(
            [
                [[0.202, 0.0, 0.0], [2.22, 0.0, 0.0], [5.35, 0.0, 0.0]],
                [[0.162, 0.0, 0.0], [2.00, 0.0, 0.0], [5.40, 0.0, 0.0]],
            ]
        )
        clashes = detect_interchain_clashes(
            coords_a,
            coords_b,
            labels_a=["A179.O", "A120.CA", "A10.CA"],
            labels_b=["B282.OE1", "B200.CA", "B48.CE"],
            max_distance_angstrom=2.2,
        )
        assert len(clashes) == 3
        assert clashes[0].frame_index == 0
        assert clashes[0].atom_a == "A179.O"
        assert clashes[0].atom_b == "B282.OE1"
        assert clashes[0].distance_angstrom == pytest.approx(2.02, abs=1e-6)
        # Frame 1 sorted by ascending distance: 0.0 Å overlap before 1.62 Å clash
        assert clashes[1].frame_index == 1
        assert clashes[1].atom_a == "A120.CA"
        assert clashes[1].distance_angstrom == pytest.approx(0.0, abs=1e-6)
        assert clashes[2].frame_index == 1
        assert clashes[2].atom_a == "A179.O"
        assert clashes[2].distance_angstrom == pytest.approx(1.62, abs=1e-6)

    def test_empty_chain_and_validation_errors(self) -> None:
        empty_a = np.zeros((1, 0, 3))
        one_b = np.zeros((1, 1, 3))
        assert detect_interchain_clashes(empty_a, one_b, [], ["B1.CA"]) == []

        with pytest.raises(ValueError, match="Atom label lengths"):
            detect_interchain_clashes(one_b, one_b, ["A1.CA", "A2.CA"], ["B1.CA"])
        with pytest.raises(ValueError, match="must be positive"):
            detect_interchain_clashes(one_b, one_b, ["A1.CA"], ["B1.CA"], max_distance_angstrom=0.0)


class TestInterfaceBSA:
    def test_bsa_conversion_from_nm2_to_angstrom2(self) -> None:
        # (100 + 120 - 170) nm² = 50 nm² total buried -> 2500 Å² one-sided interface area
        bsa = compute_interface_bsa_angstrom2(
            sasa_complex_nm2=[170.0, 180.0],
            sasa_a_nm2=[100.0, 100.0],
            sasa_b_nm2=[120.0, 120.0],
        )
        assert bsa.tolist() == pytest.approx([2500.0, 2000.0])

    def test_raises_on_shape_mismatch_or_non_finite(self) -> None:
        with pytest.raises(ValueError, match="SASA shapes must match"):
            compute_interface_bsa_angstrom2([170.0], [100.0, 100.0], [120.0])
        with pytest.raises(ValueError, match="only finite numbers"):
            compute_interface_bsa_angstrom2([np.nan], [100.0], [120.0])
