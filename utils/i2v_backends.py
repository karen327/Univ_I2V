from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_COGVIDEOX_HF_MODEL_ID = "THUDM/CogVideoX-5b-I2V"
COGVIDEOX_REGISTRY_KEYS = (
    "CogVideoX",
    "THUDM/CogVideoX-5b-I2V",
    "THUDM/CogVideoX1.5-5B-I2V",
)


@dataclass(frozen=True)
class GenerationRequest:
    image_path: Path
    output_path: Path
    prompt: str
    negative_prompt: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    num_frames: int
    width: int
    height: int
    fps: int
    device: str
    use_dynamic_cfg: bool = False
    export_quality: Optional[float] = None


@dataclass(frozen=True)
class CogVideoXBackend:
    registry_key: str
    hf_model_id: str

    def load_pipeline(self, device: str) -> Any:
        try:
            import torch
            from diffusers import CogVideoXImageToVideoPipeline
        except ImportError as exc:
            raise RuntimeError(
                "CogVideoX backend requires torch and diffusers in the active environment."
            ) from exc

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            self.hf_model_id,
            torch_dtype=torch_dtype,
        )
        pipeline = pipeline.to(device)
        return pipeline

    def generate(self, pipeline: Any, request: GenerationRequest) -> Dict[str, Any]:
        try:
            import torch
            from diffusers.utils import export_to_video, load_image
        except ImportError as exc:
            raise RuntimeError(
                "CogVideoX backend requires torch and diffusers utilities in the active environment."
            ) from exc

        with torch.no_grad():
            image = load_image(str(request.image_path)).convert("RGB")
            image = image.resize((request.width, request.height))

            generator_device = request.device if request.device.startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(request.seed)

            negative_prompt = request.negative_prompt or None
            output = pipeline(
                image=image,
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                generator=generator,
                use_dynamic_cfg=request.use_dynamic_cfg,
            )
            frames = output.frames[0]

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        export_kwargs = {
            "fps": request.fps,
        }
        if request.export_quality is not None:
            export_kwargs["quality"] = request.export_quality
        export_to_video(frames, str(request.output_path), **export_kwargs)

        return {
            "output_video_path": str(request.output_path),
            "num_frames": len(frames),
            "seed": request.seed,
            "fps": request.fps,
            "use_dynamic_cfg": request.use_dynamic_cfg,
            "export_quality": request.export_quality,
        }


BACKEND_DEFAULT_HF_MODEL_IDS = {
    "CogVideoX": DEFAULT_COGVIDEOX_HF_MODEL_ID,
    "THUDM/CogVideoX-5b-I2V": "THUDM/CogVideoX-5b-I2V",
    "THUDM/CogVideoX1.5-5B-I2V": "THUDM/CogVideoX1.5-5B-I2V",
}

BACKEND_REGISTRY = {key: "CogVideoX" for key in COGVIDEOX_REGISTRY_KEYS}


def supported_model_ids() -> list[str]:
    return sorted(BACKEND_REGISTRY.keys())


def create_backend(model_id: str, hf_model_id_override: str = "") -> CogVideoXBackend:
    if model_id not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unsupported i2v model id '{model_id}'. Supported ids: {', '.join(supported_model_ids())}"
        )

    resolved_hf_model_id = hf_model_id_override or BACKEND_DEFAULT_HF_MODEL_IDS[model_id]
    return CogVideoXBackend(
        registry_key=model_id,
        hf_model_id=resolved_hf_model_id,
    )
