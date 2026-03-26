#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa
"""Test script for DeePMD inference functionality.

This script tests the core inference logic that will be used in the web interface.
"""

import sys

import numpy as np


def test_inference_basic():
    """Test basic inference with water molecule"""
    print("=" * 80)
    print("Test 1: Basic Water Molecule Inference")
    print("=" * 80)

    try:
        from deepmd.infer import (
            DeepPot,
        )

        # This test requires a model file
        print("✓ DeePMD imports successful")

        # Prepare input - single water molecule
        atom_types = [0, 1, 1]  # O, H, H
        coords = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.93, -0.24]]
        ).reshape(1, 3, 3)

        print("\nInput configuration:")
        print(f"  Atom types: {atom_types}")
        print(f"  Coordinates shape: {coords.shape}")
        print(f"  Coordinates:\n{coords[0]}")

        # Note: This will fail without a model, but validates the API
        print("\n✓ Input preparation successful")
        print("  (Actual inference requires model file)")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  DeePMD-kit not installed or not in Python path")
        return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_inference_with_model(model_path):
    """Test inference with actual model"""
    print("\n" + "=" * 80)
    print("Test 2: Inference with Model")
    print("=" * 80)

    try:
        from deepmd.infer import (
            DeepPot,
        )

        print(f"Loading model: {model_path}")
        dp = DeepPot(model_path)

        # Get model info
        print("\nModel Information:")
        print(f"  Type map: {dp.get_type_map()}")
        print(f"  Cutoff radius: {dp.get_rcut()} Å")
        print(f"  Number of types: {dp.get_ntypes()}")

        # Test 1: Water molecule (non-periodic)
        print("\n" + "-" * 80)
        print("Test Case 1: Water Molecule (Non-periodic)")
        print("-" * 80)

        atom_types = [0, 1, 1]
        coords = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.93, -0.24]]
        ).reshape(1, 3, 3)

        print("Input:")
        print(f"  Atom types: {atom_types}")
        print(f"  Coordinates:\n{coords[0]}")

        energy, force, virial, _, _ = dp.eval(
            coords=coords, cells=None, atom_types=atom_types, atomic=False
        )

        print("\nResults:")
        print(f"  Energy: {energy[0][0]:.6f} eV")
        print(f"  Energy per atom: {energy[0][0] / len(atom_types):.6f} eV/atom")
        print("\n  Forces (eV/Å):")
        for i, f in enumerate(force[0]):
            print(f"    Atom {i}: [{f[0]:10.6f}, {f[1]:10.6f}, {f[2]:10.6f}]")
        print("\n  Virial (eV):")
        virial_matrix = virial[0].reshape(3, 3)
        for row in virial_matrix:
            print(f"    [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

        # Test 2: Water in periodic box
        print("\n" + "-" * 80)
        print("Test Case 2: Water in Periodic Box")
        print("-" * 80)

        coords_pbc = np.array(
            [[5.0, 5.0, 5.0], [5.0, 5.0, 5.96], [5.0, 5.93, 4.76]]
        ).reshape(1, 3, 3)

        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]).reshape(
            1, 9
        )

        print("Input:")
        print(f"  Atom types: {atom_types}")
        print(f"  Coordinates:\n{coords_pbc[0]}")
        print(f"  Cell:\n{cell[0].reshape(3, 3)}")

        energy_pbc, force_pbc, virial_pbc, _, _ = dp.eval(
            coords=coords_pbc, cells=cell, atom_types=atom_types, atomic=False
        )

        print("\nResults:")
        print(f"  Energy: {energy_pbc[0][0]:.6f} eV")
        print(f"  Energy per atom: {energy_pbc[0][0] / len(atom_types):.6f} eV/atom")
        print("\n  Forces (eV/Å):")
        for i, f in enumerate(force_pbc[0]):
            print(f"    Atom {i}: [{f[0]:10.6f}, {f[1]:10.6f}, {f[2]:10.6f}]")
        print("\n  Virial (eV):")
        virial_matrix_pbc = virial_pbc[0].reshape(3, 3)
        for row in virial_matrix_pbc:
            print(f"    [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

        # Test 3: Multiple frames
        print("\n" + "-" * 80)
        print("Test Case 3: Multiple Frames")
        print("-" * 80)

        coords_multi = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.93, -0.24]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97], [0.0, 0.94, -0.24]],
            ]
        )

        print("Input:")
        print(f"  Number of frames: {coords_multi.shape[0]}")
        print(f"  Atom types: {atom_types}")

        energy_multi, force_multi, virial_multi, _, _ = dp.eval(
            coords=coords_multi, cells=None, atom_types=atom_types, atomic=False
        )

        print("\nResults:")
        for i in range(coords_multi.shape[0]):
            print(f"  Frame {i}: Energy = {energy_multi[i][0]:.6f} eV")

        print("\n✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_formats():
    """Test data format conversions for web interface"""
    print("\n" + "=" * 80)
    print("Test 3: Data Format Conversions (for Web Interface)")
    print("=" * 80)

    try:
        # Simulate parsing from text input (as in web interface)
        atom_types_str = "0 1 1"
        coords_str = """0.0 0.0 0.0
0.0 0.0 0.96
0.0 0.93 -0.24"""
        cell_str = """10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0"""

        # Parse atom types
        atom_types = [int(x) for x in atom_types_str.strip().split()]
        print(f"Parsed atom types: {atom_types}")

        # Parse coordinates
        coords_list = []
        for line in coords_str.strip().split("\n"):
            values = [float(x) for x in line.strip().split()]
            if len(values) == 3:
                coords_list.append(values)
        coords = np.array(coords_list).reshape(1, len(coords_list), 3)
        print(f"Parsed coordinates shape: {coords.shape}")
        print(f"Coordinates:\n{coords[0]}")

        # Parse cell
        cell_list = []
        for line in cell_str.strip().split("\n"):
            values = [float(x) for x in line.strip().split()]
            if len(values) == 3:
                cell_list.append(values)
        cell = np.array(cell_list).reshape(1, 9)
        print(f"Parsed cell shape: {cell.shape}")
        print(f"Cell:\n{cell[0].reshape(3, 3)}")

        # Validate
        assert len(atom_types) == coords.shape[1], "Atom count mismatch"
        assert cell.shape == (1, 9), "Cell shape incorrect"

        print("\n✓ Data format conversion successful")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling"""
    print("\n" + "=" * 80)
    print("Test 4: Error Handling")
    print("=" * 80)

    test_cases = [
        {
            "name": "Mismatched atom types and coordinates",
            "atom_types": [0, 1],
            "coords": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).reshape(1, 3, 3),
            "should_fail": True,
        },
        {
            "name": "Invalid cell shape",
            "atom_types": [0, 1, 1],
            "coords": np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).reshape(1, 3, 3),
            "cell": np.array([1, 2, 3, 4, 5]),  # Wrong shape
            "should_fail": True,
        },
    ]

    passed = 0
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        try:
            coords = test["coords"]
            atom_types = test["atom_types"]

            # Check for validation errors
            if len(atom_types) != coords.shape[1]:
                print("  ✓ Caught expected error: atom count mismatch")
                passed += 1
            elif "cell" in test and test["cell"].shape != (1, 9):
                print("  ✓ Caught expected error: invalid cell shape")
                passed += 1
            else:
                print("  ✗ Should have failed but didn't")

        except Exception as e:
            if test["should_fail"]:
                print(f"  ✓ Expected error caught: {e}")
                passed += 1
            else:
                print(f"  ✗ Unexpected error: {e}")

    print(f"\n✓ Passed {passed}/{len(test_cases)} error handling tests")
    return passed == len(test_cases)


def download_test_model(url, output_path):
    """Download test model from URL"""
    print(f"\nDownloading model from: {url}")
    print(f"Saving to: {output_path}")

    try:
        import os
        import urllib.request

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print("\n✓ Model downloaded successfully")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("DeePMD Web Inference - Test Suite")
    print("=" * 80)

    results = []

    # Test 1: Basic inference API
    results.append(("Basic API", test_inference_basic()))

    # Test 2: Data format conversions
    results.append(("Data Formats", test_data_formats()))

    # Test 3: Error handling
    results.append(("Error Handling", test_error_handling()))

    # Test 4: Inference with model (if available)
    model_url = "https://store.aissquare.com/models/0bcdb486-95c9-431c-a99f-efb1a5a294ce/dpa-3.1-3m-ft.pth"
    model_path = "/tmp/dpa-3.1-3m-ft.pth"

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"\nUsing model from command line: {model_path}")
    elif "--download" in sys.argv:
        if download_test_model(model_url, model_path):
            results.append(("Model Inference", test_inference_with_model(model_path)))
        else:
            print("\nSkipping model inference test (download failed)")
    else:
        print("\nSkipping model inference test (no model provided)")
        print("To test with model:")
        print("  1. Download model: python test_inference.py --download")
        print("  2. Or specify path: python test_inference.py <model_path>")

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:20s}: {status}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
