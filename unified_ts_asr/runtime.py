"""Load the released research checkpoints without environment-specific paths."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


MODEL_FAMILIES = {
    "KwsAsrConformerDae": "text",
    "SpeakerConformerCTCOnlineEnroll": "speaker",
    "SpeakerConformerCTCEmbeddingFree": "speaker",
    "DualCueConformerCTC": "unified",
}


def _absolute(path: Optional[str], base: Path) -> Optional[str]:
    if path in (None, ""):
        return path
    value = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return str(value if value.is_absolute() else (base / value).resolve())


def resolve_config_paths(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Resolve file references in a released YAML relative to that YAML file."""
    base = Path(config_path).resolve().parent
    tokenizer = dict(config.get("tokenizer_conf") or {})
    for key in ("token_list", "seg_dict"):
        if key in tokenizer:
            tokenizer[key] = _absolute(tokenizer[key], base)
    config["tokenizer_conf"] = tokenizer
    if config.get("cmvn_file"):
        config["cmvn_file"] = _absolute(config["cmvn_file"], base)
    model_conf = dict(config.get("model_conf") or {})
    speaker_conf = dict(model_conf.get("spk_encoder_conf") or {})
    if speaker_conf.get("init_param"):
        speaker_conf["init_param"] = _absolute(speaker_conf["init_param"], base)
        model_conf["spk_encoder_conf"] = speaker_conf
    config["model_conf"] = model_conf
    return config


def load_checkpoint_state(path: str) -> Dict[str, Any]:
    """Unwrap regular and DeepSpeed checkpoint layouts used by this release."""
    import torch

    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "mp_rank_00_model_states.pt"
    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = raw
    if isinstance(raw, dict):
        for key in ("module", "state_dict", "model"):
            candidate = raw.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def _write_embedded_campplus(state: Dict[str, Any], directory: str) -> str:
    """Materialize the frozen CAM++ weights embedded in a full ASR checkpoint."""
    import torch

    prefixes = ("spk_encoder.campplus.", "spk_encoder.")
    extracted: Dict[str, Any] = {}
    for name, tensor in state.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                key = name[len(prefix):]
                if prefix == "spk_encoder." and key.startswith("campplus."):
                    key = key[len("campplus."):]
                extracted[key] = tensor
                break
    if not extracted:
        raise RuntimeError("The checkpoint does not contain embedded CAM++ parameters")
    output = str(Path(directory) / "campplus_init.pt")
    torch.save(extracted, output)
    return output


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ModelBundle:
    """A released model, tokenizer and frontend with one-call inference."""

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        self.model_dir = Path(model_dir).resolve()
        self.config_path = self.model_dir / "config.yaml"
        self.checkpoint_path = self.model_dir / "model.pt"
        if not self.config_path.is_file() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{self.model_dir} must contain config.yaml and model.pt"
            )
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        config = OmegaConf.to_container(
            OmegaConf.load(str(self.config_path)), resolve=True
        )
        self.config = resolve_config_paths(dict(config), str(self.config_path))
        self.model_name = str(self.config.get("model"))
        try:
            self.family = MODEL_FAMILIES[self.model_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported released model class: {self.model_name}") from exc

        state = load_checkpoint_state(str(self.checkpoint_path))
        self._temporary = tempfile.TemporaryDirectory(prefix="unified-ts-asr-")
        speaker_conf = dict(self.config.get("model_conf", {}).get("spk_encoder_conf") or {})
        speaker_init = speaker_conf.get("init_param")
        if self.model_name == "SpeakerConformerCTCOnlineEnroll" and (
            not speaker_init or not Path(str(speaker_init)).is_file()
        ):
            speaker_conf["init_param"] = _write_embedded_campplus(
                state, self._temporary.name
            )
            self.config["model_conf"]["spk_encoder_conf"] = speaker_conf

        self._register_release_modules()
        self.model, self.tokenizer, self.frontend = self._build_components()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Checkpoint/config mismatch: "
                f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
            )
        self.load_report = {
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
        self.model.to(self.device).eval()

    @staticmethod
    def _register_release_modules() -> None:
        for module in (
            "funasr.models.tsasr.dae_kws_encoders",
            "funasr.models.tsasr.dae_kws_asr_model",
            "funasr.models.tsasr.speaker_conformer_ctc",
            "funasr.models.tsasr.dual_cue_conformer_ctc",
        ):
            importlib.import_module(module)

    def _build_components(self) -> Tuple[Any, Any, Any]:
        from funasr.register import tables

        token_path = self.config["tokenizer_conf"]["token_list"]
        with open(token_path, encoding="utf-8") as source:
            vocab_size = len(json.load(source))
        tokenizer = tables.tokenizer_classes.get(self.config["tokenizer"])(
            **self.config["tokenizer_conf"]
        )
        frontend_conf = dict(self.config["frontend_conf"])
        frontend_conf["cmvn_file"] = self.config.get("cmvn_file")
        frontend = tables.frontend_classes.get(self.config["frontend"])(
            **frontend_conf
        )
        kwargs = dict(self.config)
        model_conf = dict(kwargs.pop("model_conf") or {})
        kwargs["vocab_size"] = vocab_size
        model_conf.update(kwargs)
        model_cls = tables.model_classes.get(self.model_name)
        if model_cls is None:
            raise KeyError(f"Model class was not registered: {self.model_name}")
        return model_cls(**model_conf), tokenizer, frontend

    @staticmethod
    def _waveform(path: str, max_seconds: float = 0.0) -> Any:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).reshape(-1)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sample_rate, 16000
            )[0]
        if max_seconds > 0:
            waveform = waveform[: int(max_seconds * 16000)]
        if waveform.numel() < 400:
            raise ValueError(f"Audio is too short: {path}")
        return waveform

    def _speaker_enrollment(self, path: str, max_seconds: float) -> Tuple[Any, Any]:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        waveform = self._waveform(path, max_seconds)
        feature = kaldi.fbank(waveform.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        length = torch.tensor([[feature.size(0)]], dtype=torch.int32)
        return feature.unsqueeze(0), length

    def _unified_enrollment(self, path: str, max_seconds: float) -> Tuple[Any, Any]:
        import torch
        from funasr.utils.load_utils import extract_fbank

        feature, lengths = extract_fbank(
            [self._waveform(path, max_seconds)],
            data_type="sound",
            frontend=self.frontend,
            is_final=True,
        )
        return feature, lengths.reshape(-1, 1).to(torch.int32)

    def transcribe(
        self,
        audio: str,
        text_cue: Optional[str] = None,
        enrollment_audio: Optional[str] = None,
        enrollment_max_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """Transcribe one mixture using the cues required by the model family."""
        import torch

        conditioning: Dict[str, Any] = {}
        if self.family == "text":
            if not text_cue:
                raise ValueError("A text-cue model requires --text-cue")
            conditioning["cue_text"] = [text_cue]
        elif self.family == "speaker":
            if not enrollment_audio:
                raise ValueError("A speaker-cue model requires --enrollment-audio")
            ref, lengths = self._speaker_enrollment(
                enrollment_audio, enrollment_max_seconds
            )
            conditioning.update(ref_speech=ref, ref_speech_lengths=lengths)
        else:
            if text_cue:
                conditioning["cue_text"] = [text_cue]
            if enrollment_audio:
                ref, lengths = self._unified_enrollment(
                    enrollment_audio, enrollment_max_seconds
                )
                conditioning.update(ref_speech=ref, ref_speech_lengths=lengths)
            if not conditioning:
                raise ValueError(
                    "A unified model requires --text-cue, --enrollment-audio, or both"
                )
        key = Path(audio).stem
        with torch.inference_mode():
            outputs, metadata = self.model.inference(
                data_in=[str(Path(audio).resolve())],
                key=[key],
                tokenizer=self.tokenizer,
                frontend=self.frontend,
                device=self.device,
                **conditioning,
            )
        output = outputs[0] if outputs else {}
        result = {
            "key": key,
            "text": output.get("text", "") if isinstance(output, dict) else str(output),
            "model": self.model_dir.name,
            "family": self.family,
            "device": self.device,
        }
        if isinstance(output, dict):
            for name in ("token", "kws_hit", "frame_span", "no_kws", "no_tgt_spk"):
                if name in output:
                    result[name] = _jsonable(output[name])
        if isinstance(metadata, dict):
            result["timing"] = _jsonable(metadata)
        return result

