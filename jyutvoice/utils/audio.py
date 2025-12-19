import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class LinearSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, pad, center, pad_mode):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.center = center
        self.pad_mode = pad_mode

        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1), (self.pad, self.pad), self.pad_mode
        ).squeeze(1)
        spec = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.pad_mode,
            False,
            True,
            True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        f_min,
        f_max,
        pad,
        n_mels,
        center,
        pad_mode,
        mel_scale,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.mel_scale = mel_scale

        self.spectrogram = LinearSpectrogram(
            n_fft, win_length, hop_length, pad, center, pad_mode
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels, sample_rate, f_min, f_max, (n_fft // 2) + 1, mel_scale, mel_scale
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor) -> Tensor:
        linear_spec = self.spectrogram(x)
        x = self.mel_scale(linear_spec)
        x = self.compress(x)
        return x


def load_and_resample_audio(audio_path, target_sr, device="cpu") -> Tensor | None:
    try:
        y, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(str(e))
        return None

    y.to(device)
    # Convert to mono
    if y.size(0) > 1:
        y = y[0, :].unsqueeze(0)  # shape: [2, time] -> [time] -> [1, time]

    # resample audio to target sample_rate
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)
    return y


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
