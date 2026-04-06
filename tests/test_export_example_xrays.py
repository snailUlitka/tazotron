from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

VARIANT_004_SHELL_DEPTH = 0.22
VARIANT_005_SHELL_DEPTH = 0.24


def _load_export_example_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "export_example_xrays.py"
    spec = importlib.util.spec_from_file_location("export_example_xrays", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_variants_keep_reference_004_and_005_values() -> None:
    module = _load_export_example_module()
    args = argparse.Namespace(**vars(module.parse_args.__globals__["DEFAULT_EXAMPLE_NECRO_CONFIG"]))
    args.case_dir = Path("/tmp/case")
    args.output_dir = None
    args.device = "cpu"
    args.save_tiff = False
    args.batch_presets = True

    variants = module.batch_variants(args)

    variant_004 = variants[3]
    variant_005 = variants[4]
    assert variant_004.effect_strength_range == (1.0, 1.1)
    assert variant_004.collapse_strength_range == (1.18, 1.34)
    assert variant_004.shell_depth_jitter_ratio == VARIANT_004_SHELL_DEPTH
    assert variant_005.effect_strength_range == (1.1, 1.3)
    assert variant_005.collapse_strength_range == (1.24, 1.42)
    assert variant_005.shell_depth_jitter_ratio == VARIANT_005_SHELL_DEPTH


def test_cli_accepts_both_and_target_head_weights(monkeypatch) -> None:
    module = _load_export_example_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_example_xrays.py",
            "--case-dir",
            "/tmp/case",
            "--target-head",
            "both",
            "--target-head-weights",
            "0.3,0.4,0.3",
        ],
    )

    args = module.parse_args()

    assert args.target_head == "both"
    assert args.target_head_weights == (0.3, 0.4, 0.3)
