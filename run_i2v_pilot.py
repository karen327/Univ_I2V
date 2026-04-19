from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from utils.i2v_backends import GenerationRequest, create_backend, supported_model_ids


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal Stage B image-to-video pilot directly from sample_subset_manifest.json."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest_path", default="", help="Path to sample_subset_manifest.json.")
    input_group.add_argument(
        "--pilot_dir",
        default="",
        help="Pilot directory containing sample_subset_manifest.json.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device used for CogVideoX generation.")
    parser.add_argument(
        "--hf_model_id",
        default="",
        help="Optional Hugging Face model id override for the CogVideoX backend.",
    )
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS.")
    parser.add_argument(
        "--negative_prompt",
        default="",
        help="Optional fixed negative prompt shared by clean / cand_a / cand_b.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        help="Enable CogVideoX dynamic CFG. Off by default to preserve the legacy runner behavior.",
    )
    parser.add_argument(
        "--export_quality",
        type=float,
        default=-1,
        help="Optional MP4 export quality override. Use a negative value to keep the legacy export default.",
    )
    parser.add_argument(
        "--prompt_override_json",
        default="",
        help=(
            "Optional JSON file mapping sample_id or image_id to prompt text. "
            "When omitted, all assets keep using protocol.prompt."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing video outputs and sample metadata.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="Only process the first N samples in manifest order. Use 0 for all samples.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate manifest fields, path resolution, and planned_outputs alignment without loading the backend.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_path:
        return Path(args.manifest_path).expanduser().resolve()
    return (Path(args.pilot_dir).expanduser() / "sample_subset_manifest.json").resolve()


def resolve_input_path(path_str: str, manifest_path: Path) -> Path:
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        if raw_path.is_file():
            return raw_path
        raise FileNotFoundError(f"Input path does not exist: {raw_path}")

    candidates = [
        (Path.cwd() / raw_path).resolve(),
        (REPO_ROOT / raw_path).resolve(),
        (manifest_path.parent / raw_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    rendered = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Input path '{path_str}' was not found. Tried: {rendered}")


def resolve_output_path(path_str: str, manifest_path: Path) -> Path:
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    repo_candidate = (REPO_ROOT / raw_path).resolve()
    manifest_candidate = (manifest_path.parent / raw_path).resolve()
    candidates = [cwd_candidate, repo_candidate, manifest_candidate]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in candidates:
        if candidate.parent.exists():
            return candidate
    return repo_candidate


def load_prompt_overrides(path_str: str) -> Dict[str, str]:
    if not path_str:
        return {}

    override_path = Path(path_str).expanduser().resolve()
    if not override_path.is_file():
        raise FileNotFoundError(f"Prompt override JSON not found: {override_path}")

    payload = load_json(override_path)
    if not isinstance(payload, dict):
        raise ValueError("prompt_override_json must contain a top-level JSON object.")

    normalized: Dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("prompt_override_json must map string keys to string prompt values.")
        normalized[key] = value
    return normalized


def resolve_prompt_for_sample(
    sample: Dict[str, Any],
    protocol: Dict[str, Any],
    prompt_overrides: Dict[str, str],
) -> str:
    sample_id = str(sample.get("sample_id", "")).strip()
    image_id = str(sample.get("image_id", "")).strip()
    if sample_id and sample_id in prompt_overrides:
        return prompt_overrides[sample_id]
    if image_id and image_id in prompt_overrides:
        return prompt_overrides[image_id]
    if "default" in prompt_overrides:
        return prompt_overrides["default"]
    return str(protocol["prompt"])


def require_fields(payload: Dict[str, Any], field_names: Iterable[str], prefix: str) -> None:
    for field_name in field_names:
        if field_name not in payload:
            raise KeyError(f"Missing required field '{prefix}{field_name}'")


def normalize_samples(manifest: Dict[str, Any], limit_samples: int) -> list[Dict[str, Any]]:
    samples = manifest["samples"]
    if limit_samples <= 0:
        return samples
    return samples[:limit_samples]


def validate_protocol(protocol: Dict[str, Any]) -> None:
    require_fields(
        protocol,
        [
            "primary_candidate_id",
            "secondary_candidate_id",
            "candidate_config_ids",
            "i2v_model_ids",
            "prompt",
            "seed",
            "num_inference_steps",
            "guidance_scale",
            "num_frames",
            "frame_resolution",
        ],
        "protocol.",
    )
    require_fields(protocol["frame_resolution"], ["width", "height"], "protocol.frame_resolution.")

    model_ids = protocol["i2v_model_ids"]
    if not isinstance(model_ids, list) or not model_ids:
        raise ValueError("protocol.i2v_model_ids must be a non-empty list.")

    unsupported = [model_id for model_id in model_ids if model_id not in supported_model_ids()]
    if unsupported:
        raise ValueError(
            "This minimal runner only supports CogVideoX ids. "
            f"Unsupported model ids: {unsupported}. Supported ids: {supported_model_ids()}"
        )


def build_assets_for_sample(
    sample: Dict[str, Any],
    protocol: Dict[str, Any],
    model_id: str,
    manifest_path: Path,
) -> list[Dict[str, Any]]:
    primary_candidate_id = protocol["primary_candidate_id"]
    secondary_candidate_id = protocol["secondary_candidate_id"]

    require_fields(sample, ["sample_id", "clean_image_path", "candidates", "planned_outputs"], "sample.")
    require_fields(sample["candidates"], [primary_candidate_id, secondary_candidate_id], "sample.candidates.")
    require_fields(
        sample["candidates"][primary_candidate_id],
        ["protected_image_path"],
        f"sample.candidates[{primary_candidate_id}].",
    )
    require_fields(
        sample["candidates"][secondary_candidate_id],
        ["protected_image_path"],
        f"sample.candidates[{secondary_candidate_id}].",
    )
    if model_id not in sample["planned_outputs"]:
        raise KeyError(
            f"Sample '{sample['sample_id']}' is missing planned_outputs for model id '{model_id}'."
        )

    planned_outputs = sample["planned_outputs"][model_id]
    require_fields(
        planned_outputs,
        ["clean_video_path", primary_candidate_id, secondary_candidate_id],
        f"sample.planned_outputs[{model_id}].",
    )

    return [
        {
            "asset_id": "clean",
            "input_image_path": resolve_input_path(sample["clean_image_path"], manifest_path),
            "output_video_path": resolve_output_path(planned_outputs["clean_video_path"], manifest_path),
        },
        {
            "asset_id": primary_candidate_id,
            "input_image_path": resolve_input_path(
                sample["candidates"][primary_candidate_id]["protected_image_path"],
                manifest_path,
            ),
            "output_video_path": resolve_output_path(planned_outputs[primary_candidate_id], manifest_path),
        },
        {
            "asset_id": secondary_candidate_id,
            "input_image_path": resolve_input_path(
                sample["candidates"][secondary_candidate_id]["protected_image_path"],
                manifest_path,
            ),
            "output_video_path": resolve_output_path(planned_outputs[secondary_candidate_id], manifest_path),
        },
    ]


def validate_manifest(
    manifest: Dict[str, Any],
    manifest_path: Path,
    limit_samples: int,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]], list[Dict[str, Any]]]:
    require_fields(manifest, ["protocol", "samples"], "")
    protocol = manifest["protocol"]
    validate_protocol(protocol)

    samples = normalize_samples(manifest, limit_samples)
    validation_rows = []
    for sample in samples:
        row = {
            "sample_id": sample.get("sample_id"),
            "image_id": sample.get("image_id"),
            "models": [],
        }
        for model_id in protocol["i2v_model_ids"]:
            assets = build_assets_for_sample(sample, protocol, model_id, manifest_path)
            row["models"].append(
                {
                    "model_id": model_id,
                    "assets": [
                        {
                            "asset_id": asset["asset_id"],
                            "input_image_path": str(asset["input_image_path"]),
                            "output_video_path": str(asset["output_video_path"]),
                        }
                        for asset in assets
                    ],
                }
            )
        validation_rows.append(row)
    return protocol, samples, validation_rows


def print_validation_summary(
    manifest_path: Path,
    protocol: Dict[str, Any],
    validation_rows: list[Dict[str, Any]],
) -> None:
    print(f"Manifest: {manifest_path}")
    print(f"Samples validated: {len(validation_rows)}")
    print(f"Model ids: {protocol['i2v_model_ids']}")
    print(
        "Protocol: "
        f"seed={protocol['seed']}, "
        f"steps={protocol['num_inference_steps']}, "
        f"guidance={protocol['guidance_scale']}, "
        f"frames={protocol['num_frames']}, "
        f"resolution={protocol['frame_resolution']['width']}x{protocol['frame_resolution']['height']}"
    )
    for row in validation_rows:
        for model_row in row["models"]:
            outputs = ", ".join(
                f"{asset['asset_id']} -> {asset['output_video_path']}"
                for asset in model_row["assets"]
            )
            print(f"- {row['sample_id']} [{model_row['model_id']}] {outputs}")


def build_generation_request(
    asset: Dict[str, Any],
    sample: Dict[str, Any],
    protocol: Dict[str, Any],
    prompt_overrides: Dict[str, str],
    args: argparse.Namespace,
) -> GenerationRequest:
    export_quality = None if args.export_quality < 0 else float(args.export_quality)
    return GenerationRequest(
        image_path=asset["input_image_path"],
        output_path=asset["output_video_path"],
        prompt=resolve_prompt_for_sample(sample, protocol, prompt_overrides),
        negative_prompt=args.negative_prompt,
        seed=int(protocol["seed"]),
        num_inference_steps=int(protocol["num_inference_steps"]),
        guidance_scale=float(protocol["guidance_scale"]),
        num_frames=int(protocol["num_frames"]),
        width=int(protocol["frame_resolution"]["width"]),
        height=int(protocol["frame_resolution"]["height"]),
        fps=int(args.fps),
        device=args.device,
        use_dynamic_cfg=bool(args.use_dynamic_cfg),
        export_quality=export_quality,
    )


def build_sample_metadata(
    sample: Dict[str, Any],
    protocol: Dict[str, Any],
    model_id: str,
    backend_hf_model_id: str,
    args: argparse.Namespace,
    asset_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "image_id": sample.get("image_id"),
        "source_bucket": sample.get("source_bucket"),
        "source_class_index": sample.get("source_class_index"),
        "source_class_text": sample.get("source_class_text"),
        "i2v_model_id": model_id,
        "hf_model_id": backend_hf_model_id,
        "device": args.device,
        "fps": args.fps,
        "negative_prompt": args.negative_prompt,
        "protocol": {
            "prompt": protocol["prompt"],
            "seed": protocol["seed"],
            "num_inference_steps": protocol["num_inference_steps"],
            "guidance_scale": protocol["guidance_scale"],
            "num_frames": protocol["num_frames"],
            "frame_resolution": protocol["frame_resolution"],
            "primary_candidate_id": protocol["primary_candidate_id"],
            "secondary_candidate_id": protocol["secondary_candidate_id"],
        },
        "conditions_match_except_input": True,
        "assets": asset_records,
        "completed_at": utc_now_iso(),
    }


def process_sample_for_model(
    sample: Dict[str, Any],
    protocol: Dict[str, Any],
    model_id: str,
    manifest_path: Path,
    prompt_overrides: Dict[str, str],
    backend: Any,
    pipeline: Any,
    args: argparse.Namespace,
    run_log_path: Path,
) -> Dict[str, Any]:
    assets = build_assets_for_sample(sample, protocol, model_id, manifest_path)
    sample_output_dir = assets[0]["output_video_path"].parent
    metadata_path = sample_output_dir / "run_metadata.json"

    asset_records: Dict[str, Dict[str, Any]] = {}
    sample_errors = []

    for asset in assets:
        request = build_generation_request(asset, sample, protocol, prompt_overrides, args)
        output_path = request.output_path
        status = "generated"
        started_at = utc_now_iso()

        if output_path.exists() and not args.overwrite:
            status = "skipped_existing"
            record = {
                "input_image_path": str(request.image_path),
                "output_video_path": str(output_path),
                "status": status,
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "seed": request.seed,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "use_dynamic_cfg": request.use_dynamic_cfg,
                "export_quality": request.export_quality,
            }
            asset_records[asset["asset_id"]] = record
            append_jsonl(
                run_log_path,
                {
                    "event": "asset_complete",
                    "sample_id": sample["sample_id"],
                    "i2v_model_id": model_id,
                    "asset_id": asset["asset_id"],
                    **record,
                },
            )
            print(f"[skip] {sample['sample_id']} [{model_id}] {asset['asset_id']} -> {output_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {sample['sample_id']} [{model_id}] {asset['asset_id']} -> {output_path}")
        started = time.time()
        try:
            backend_result = backend.generate(pipeline, request)
        except Exception as exc:  # noqa: BLE001
            error_payload = {
                "input_image_path": str(request.image_path),
                "output_video_path": str(output_path),
                "status": "failed",
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "error": str(exc),
            }
            asset_records[asset["asset_id"]] = error_payload
            sample_errors.append(
                {
                    "asset_id": asset["asset_id"],
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            append_jsonl(
                run_log_path,
                {
                    "event": "asset_failed",
                    "sample_id": sample["sample_id"],
                    "i2v_model_id": model_id,
                    "asset_id": asset["asset_id"],
                    **error_payload,
                },
            )
            continue

        elapsed = round(time.time() - started, 4)
        record = {
            "input_image_path": str(request.image_path),
            "output_video_path": str(output_path),
            "status": status,
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "elapsed_seconds": elapsed,
            "seed": request.seed,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "num_frames": request.num_frames,
            "frame_resolution": {"width": request.width, "height": request.height},
            "fps": request.fps,
            "use_dynamic_cfg": request.use_dynamic_cfg,
            "export_quality": request.export_quality,
            "backend_result": backend_result,
        }
        asset_records[asset["asset_id"]] = record
        append_jsonl(
            run_log_path,
            {
                "event": "asset_complete",
                "sample_id": sample["sample_id"],
                "i2v_model_id": model_id,
                "asset_id": asset["asset_id"],
                **record,
            },
        )

    sample_metadata = build_sample_metadata(
        sample=sample,
        protocol=protocol,
        model_id=model_id,
        backend_hf_model_id=backend.hf_model_id,
        args=args,
        asset_records=asset_records,
    )
    if sample_errors:
        sample_metadata["errors"] = sample_errors

    if args.overwrite or not metadata_path.exists() or sample_errors or asset_records:
        dump_json(metadata_path, sample_metadata)

    return {
        "sample_id": sample["sample_id"],
        "i2v_model_id": model_id,
        "metadata_path": str(metadata_path),
        "num_assets": len(asset_records),
        "num_failed_assets": len(sample_errors),
        "asset_statuses": {
            asset_id: record["status"] for asset_id, record in asset_records.items()
        },
    }


def main() -> int:
    args = parse_args()
    manifest_path = resolve_manifest_path(args)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    pilot_dir = manifest_path.parent
    manifest = load_json(manifest_path)
    protocol, samples, validation_rows = validate_manifest(
        manifest,
        manifest_path,
        args.limit_samples,
    )
    prompt_overrides = load_prompt_overrides(args.prompt_override_json)

    print_validation_summary(manifest_path, protocol, validation_rows)
    if args.dry_run:
        print("Dry-run completed. No backend was loaded and no videos were generated.")
        return 0

    run_started_at = utc_now_iso()
    run_log_path = pilot_dir / "run_i2v_pilot_log.jsonl"
    run_summary_path = pilot_dir / "run_i2v_pilot_summary.json"
    append_jsonl(
        run_log_path,
        {
            "event": "run_started",
            "started_at": run_started_at,
            "manifest_path": str(manifest_path),
            "pilot_dir": str(pilot_dir),
            "limit_samples": args.limit_samples,
            "device": args.device,
            "fps": args.fps,
            "negative_prompt": args.negative_prompt,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "export_quality": None if args.export_quality < 0 else args.export_quality,
            "prompt_override_json": args.prompt_override_json,
            "hf_model_id_override": args.hf_model_id,
        },
    )

    per_model_backends: Dict[str, Any] = {}
    per_model_pipelines: Dict[str, Any] = {}
    for model_id in protocol["i2v_model_ids"]:
        backend = create_backend(model_id, hf_model_id_override=args.hf_model_id)
        per_model_backends[model_id] = backend
        per_model_pipelines[model_id] = backend.load_pipeline(args.device)

    sample_results = []
    total_failed_assets = 0
    for sample in samples:
        for model_id in protocol["i2v_model_ids"]:
            result = process_sample_for_model(
                sample=sample,
                protocol=protocol,
                model_id=model_id,
                manifest_path=manifest_path,
                prompt_overrides=prompt_overrides,
                backend=per_model_backends[model_id],
                pipeline=per_model_pipelines[model_id],
                args=args,
                run_log_path=run_log_path,
            )
            sample_results.append(result)
            total_failed_assets += result["num_failed_assets"]

    run_summary = {
        "manifest_path": str(manifest_path),
        "pilot_dir": str(pilot_dir),
        "started_at": run_started_at,
        "completed_at": utc_now_iso(),
        "device": args.device,
        "fps": args.fps,
        "negative_prompt": args.negative_prompt,
        "use_dynamic_cfg": args.use_dynamic_cfg,
        "export_quality": None if args.export_quality < 0 else args.export_quality,
        "prompt_override_json": args.prompt_override_json,
        "hf_model_id_override": args.hf_model_id,
        "protocol": protocol,
        "num_samples_requested": len(samples),
        "num_model_runs": len(sample_results),
        "num_failed_assets": total_failed_assets,
        "sample_results": sample_results,
    }
    dump_json(run_summary_path, run_summary)
    append_jsonl(
        run_log_path,
        {
            "event": "run_completed",
            "completed_at": run_summary["completed_at"],
            "num_samples_requested": run_summary["num_samples_requested"],
            "num_model_runs": run_summary["num_model_runs"],
            "num_failed_assets": total_failed_assets,
            "run_summary_path": str(run_summary_path),
        },
    )

    if total_failed_assets > 0:
        print(f"Run finished with {total_failed_assets} failed asset(s). See {run_log_path}")
        return 1

    print(f"Run finished successfully. Summary: {run_summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
