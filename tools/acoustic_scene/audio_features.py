"""Shared waveform and spectral frontend for training, export, and parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio


CONFIG_PATH = Path(__file__).with_name("model_config.json")


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_waveform(path: Path, sample_rate: int, window_samples: int) -> np.ndarray:
    samples, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if source_rate != sample_rate:
        mono = torchaudio.functional.resample(
            torch.from_numpy(mono), source_rate, sample_rate
        ).numpy()
    if mono.size > window_samples:
        mono = mono[:window_samples]
    elif mono.size < window_samples:
        mono = np.pad(mono, (0, window_samples - mono.size))
    return np.asarray(mono, dtype=np.float32)


class WaveformFrontend(nn.Module):
    """Official EfficientAT-compatible 32 kHz log-Mel frontend."""

    def __init__(self, config: dict[str, object], augment: bool = False) -> None:
        super().__init__()
        self.sample_rate = int(config["sample_rate"])
        self.n_fft = int(config["n_fft"])
        self.win_length = int(config["win_length"])
        self.hop_length = int(config["hop_length"])
        self.n_mels = int(config["n_mels"])
        self.f_min = float(config["f_min"])
        self.f_max = float(config["f_max"])
        self.augment = augment
        self.register_buffer(
            "preemphasis",
            torch.tensor([[[-0.97, 1.0]]], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length, periodic=False),
            persistent=False,
        )
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sample_rate,
            self.f_min,
            self.f_max,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), value=0.0)
        self.register_buffer("mel_basis", mel_basis.float())
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(16, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(64, iid_masks=True)

    def power_spectrogram(self, waveforms: torch.Tensor) -> torch.Tensor:
        emphasized = torch.nn.functional.conv1d(
            waveforms.unsqueeze(1), self.preemphasis
        ).squeeze(1)
        spectrum = torch.stft(
            emphasized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        )
        return spectrum.abs().square()

    def mel_from_power(self, power: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(self.mel_basis, power)
        mel = torch.log(mel + 1e-5)
        mel = (mel + 4.5) / 5.0
        if self.training and self.augment:
            mel = self.frequency_mask(mel)
            mel = self.time_mask(mel)
        return mel

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.mel_from_power(self.power_spectrogram(waveforms))


class PowerToLogMel(nn.Module):
    """ONNX-friendly half of the frontend; native code supplies FFT power."""

    def __init__(self, mel_basis: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mel_basis", mel_basis.detach().clone().float())

    def forward(self, power: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(self.mel_basis, power)
        return (torch.log(mel + 1e-5) + 4.5) / 5.0
