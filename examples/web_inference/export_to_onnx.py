#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa
"""Export DeePMD-kit PyTorch model to ONNX format for web inference.

This script exports a trained DeePMD-kit model to ONNX format, which can be
used for browser-based inference with ONNX Runtime Web.
"""

import argparse
import sys

import numpy as np
import torch


def export_model_to_onnx(model_path: str, output_path: str, natoms: int = 3):
    """Export DeePMD model to ONNX format.

    Args:
        model_path: Path to the .pt or .pth model file
        output_path: Path where to save the ONNX model
        natoms: Number of atoms for the example (default: 3 for water)
    """
    print(f"Loading model from: {model_path}")

    try:
        from deepmd.pt.model.model import (
            get_model,
        )
        from deepmd.pt.utils.utils import (
            to_numpy_array,
            to_torch_tensor,
        )
    except ImportError:
        print("Error: DeePMD-kit PyTorch backend not installed")
        print("Please install: pip install deepmd-kit[torch]")
        sys.exit(1)

    # Load the model
    try:
        if model_path.endswith(".pth"):
            # JIT scripted model
            model = torch.jit.load(model_path)
        else:
            # Regular checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")
            model_params = checkpoint["model"]["_extra_state"]["model_params"]
            model = get_model(model_params)
            model.load_state_dict(checkpoint["model"])
            model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Model loaded successfully")
    print(f"Model type: {type(model)}")

    # Create dummy inputs for tracing
    # For water molecule (3 atoms)
    coord = torch.zeros(1, natoms, 3, dtype=torch.float64)
    atype = torch.zeros(natoms, dtype=torch.int32)
    box = torch.zeros(1, 3, 3, dtype=torch.float64)

    # Example: water molecule coordinates
    if natoms == 3:
        coord[0] = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.93, -0.24]],
            dtype=torch.float64,
        )
        atype = torch.tensor([0, 1, 1], dtype=torch.int32)  # O, H, H

    print(f"\nDummy input shapes:")
    print(f"  coord: {coord.shape}")
    print(f"  atype: {atype.shape}")
    print(f"  box: {box.shape}")

    # Set model to evaluation mode
    model.eval()

    # Export to ONNX
    print(f"\nExporting to ONNX format: {output_path}")

    try:
        torch.onnx.export(
            model,
            (coord, atype, box),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["coord", "atype", "box"],
            output_names=["energy", "force", "virial"],
            dynamic_axes={
                "coord": {0: "batch_size", 1: "natoms"},
                "atype": {0: "natoms"},
                "box": {0: "batch_size"},
                "energy": {0: "batch_size"},
                "force": {0: "batch_size", 1: "natoms"},
                "virial": {0: "batch_size"},
            },
        )
        print(f"✓ Model exported successfully to: {output_path}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        print("\nNote: ONNX export may not work for all DeePMD models.")
        print("The model architecture must be compatible with ONNX operations.")
        sys.exit(1)

    # Verify the exported model
    print("\nVerifying exported ONNX model...")
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        print("\nModel inputs:")
        for input in onnx_model.graph.input:
            print(f"  - {input.name}: {input.type}")

        print("\nModel outputs:")
        for output in onnx_model.graph.output:
            print(f"  - {output.name}: {output.type}")

    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: Verification failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export DeePMD-kit model to ONNX format for web inference"
    )
    parser.add_argument(
        "model", type=str, help="Path to the input model file (.pt or .pth)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.onnx",
        help="Path to the output ONNX file (default: model.onnx)",
    )
    parser.add_argument(
        "-n",
        "--natoms",
        type=int,
        default=3,
        help="Number of atoms for dummy input (default: 3)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DeePMD-kit Model Export to ONNX")
    print("=" * 80)

    export_model_to_onnx(args.model, args.output, args.natoms)

    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Copy {args.output} to examples/web_inference/")
    print(f"2. Open index.html in a web browser")
    print(f"3. The model will be loaded automatically")
    print(f"\nNote: Make sure the model is accessible via HTTP when serving the page.")


if __name__ == "__main__":
    main()
