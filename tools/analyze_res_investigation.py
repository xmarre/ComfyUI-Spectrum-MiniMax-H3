"""Compare full latent checkpoints from two Spectrum RES investigation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_spectrum_h3.bsa_transition_probe import delta
from comfyui_spectrum_h3.res_investigation import _streams


def compare_reports(reference_path, candidate_path):
    paths = [Path(reference_path), Path(candidate_path)]
    reference, candidate = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    for report in (reference, candidate):
        if report.get("schema") != "spectrum_h3_res_investigation_v1":
            raise ValueError("unsupported RES investigation report schema")
        if not report.get("valid_trace"):
            raise ValueError(
                "both RES investigation traces must be complete and verified"
            )
    for field in (
        "seed",
        "sigmas",
        "sigma_dtype",
        "latent_shapes",
        "native_source_sha256",
        "masked",
    ):
        if reference.get(field) != candidate.get(field):
            raise ValueError(
                f"RES reports differ in {field}; this is not a matched comparison"
            )
    checkpoints = []
    for path, report in zip(paths, (reference, candidate), strict=True):
        filename = report["tensor_file"]
        if Path(filename).name != filename:
            raise ValueError("tensor_file must be a sibling filename")
        checkpoints.append(load_file(str(path.with_name(filename))))
    ref_tensors, cand_tensors = checkpoints
    measurements = {}
    for key in sorted(ref_tensors.keys() & cand_tensors.keys()):
        a, b = ref_tensors[key], cand_tensors[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"RES checkpoint shapes/dtypes differ at {key}")
        pieces = {}
        streams = (
            (("value", a),)
            if key.startswith("final_")
            else _streams(a, reference["latent_shapes"])
        )
        other = (
            (("value", b),)
            if key.startswith("final_")
            else _streams(b, candidate["latent_shapes"])
        )
        for (name, x), (_, y) in zip(streams, other, strict=True):
            pieces[name] = {
                "bitwise_equal": torch.equal(
                    x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
                ),
                **delta(x, y),
            }
        measurements[key] = pieces
    selected = candidate["settings"]["selected_step"]
    ref_steps = {row["step"]: row for row in reference["steps"]}
    cand_steps = {row["step"]: row for row in candidate["steps"]}
    matching_prefix = all(
        i in ref_steps
        and i in cand_steps
        and ref_steps[i]["input"] == cand_steps[i]["input"]
        and (i == selected or ref_steps[i]["denoised"] == cand_steps[i]["denoised"])
        for i in range(selected + 1)
    )
    return {
        "schema": "spectrum_h3_res_comparison_v1",
        "reference": str(paths[0]),
        "candidate": str(paths[1]),
        "same_config": reference["config"] == candidate["config"],
        "same_phase": reference["phase"] == candidate["phase"],
        "identical_prefix_through_selected_input": matching_prefix,
        "reference_selected_status": reference["selected_status"],
        "candidate_selected_status": candidate["selected_status"],
        "candidate_same_state_errors": [
            row["same_state_forecast_error"]
            for row in candidate["steps"]
            if "same_state_forecast_error" in row
        ],
        "checkpoints": measurements,
        "scope": "Full checkpoint tensor deltas; decoded media quality and latency require separate evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_reports(args.reference, args.candidate)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
