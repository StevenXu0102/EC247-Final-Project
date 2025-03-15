# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


##FFT
@dataclass
class FFTTransform:
    """
    A generic wrapper that applies a fast Fourier transformation on the output 
    of any given transform. You can also choose to perform an inverse FFT to return 
    the data back to the time domain.

    Args:
        transform (Callable): The transform to wrap.
        fft_dim (int): Dimension along which to compute the FFT.
        n (int, optional): FFT length. If None, the FFT is computed with the full size.
        apply_inverse (bool): If True, applies the inverse FFT (and returns the real part).
    """
    transform: Transform[torch.Tensor, torch.Tensor]
    fft_dim: int = 0
    n: int = None
    apply_inverse: bool = False

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.transform(tensor)
        fft_out = torch.fft.fft(out, n=self.n, dim=self.fft_dim)
        if self.apply_inverse:
            return torch.fft.ifft(fft_out, n=self.n, dim=self.fft_dim).real
        return fft_out


@dataclass
class FFTLogSpectrogram:
    """
    Computes a log-scaled spectrogram using a manual FFT implementation.
    This transform manually splits the input tensor into overlapping windows,
    applies a Hann window, computes the FFT, and then calculates the power 
    spectrogram (magnitude squared) with log10 scaling.

    Args:
        n_fft (int): Size of the FFT window.
        hop_length (int): Number of samples to step between successive windows.
    """
    n_fft: int = 64
    hop_length: int = 16

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        dtype = tensor.dtype

        window = torch.hann_window(self.n_fft, device=device, dtype=dtype)
        
        frames = tensor.unfold(dimension=0, size=self.n_fft, step=self.hop_length)
        
        frames = frames * window
        
        spec = torch.fft.rfft(frames, n=self.n_fft)
        spec_power = spec.abs() ** 2
        logspec = torch.log10(spec_power + 1e-6)
        return logspec

##Wavelet transformation
from dataclasses import dataclass
import torch
import numpy as np
import pywt

@dataclass
class LogWaveletSpectrogram:
    """
    Creates a log10-scaled continuous wavelet transform (CWT) representation from an EMG signal.
    In the case of a multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned representation is of shape 
    (T_new, ..., scales), where T_new is the downsampled time dimension (using hop_length)
    and 'scales' corresponds to the number of wavelet scales.
    
    Args:
        wavelet (str): The wavelet to use for the CWT (default: 'morl').
        scales (Optional[Iterable[int]]): Sequence of scales to use for the CWT. If None,
            defaults to np.arange(1, 34) (i.e. 33 scales).
        hop_length (int): Downsampling factor along the time dimension (default: 16).
    """
    wavelet: str = 'morl'
    scales: np.ndarray = None
    hop_length: int = 16

    def __post_init__(self) -> None:
        if self.scales is None:
            self.scales = np.arange(1, 34)  

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)
        orig_shape = x.shape[:-1]
        T = x.shape[-1]
        x_flat = x.reshape(-1, T)

        outputs = []
        for signal in x_flat:
            signal_np = signal.cpu().numpy()
            coeffs, _ = pywt.cwt(signal_np, self.scales, self.wavelet)
            coeffs_ds = coeffs[:, ::self.hop_length]  
            log_coeffs = np.log10(np.abs(coeffs_ds) + 1e-6)
            outputs.append(log_coeffs)

        outputs = np.stack(outputs, axis=0) 
        new_shape = orig_shape + (len(self.scales), outputs.shape[-1])
        outputs = outputs.reshape(new_shape)
        outputs = torch.tensor(outputs)
        outputs = outputs.movedim(-1, 0) 
        return outputs

##Add noise
@dataclass
class GaussianNoise:
    """
    Adds Gaussian noise to a tensor.

    The noise is sampled from a Gaussian (normal) distribution with a specified mean
    and standard deviation, and is added element-wise to the input tensor. This can be
    used for data augmentation on time-series, spectrograms, or any other tensor-based data.

    Args:
        mean (float): Mean of the Gaussian noise. Default is 0.0.
        std (float): Standard deviation of the Gaussian noise. Default is 1.0.
        p (float): Probability of applying the noise. Default is 1.0 (always applied).
    """
    mean: float = 0.0
    std: float = 1.0
    p: float = 1.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

##Parallel that uses both FFT and regular    
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Optional
import torch

@dataclass
class SkewAverages:
    """
    Applies multiple transforms in parallel on the same input and computes a weighted average of their outputs.
    This allows you to skew the contributions of each branch during backpropagation.

    Args:
        transforms (Sequence[Callable[[Any], Any]]): A list of transforms to apply in parallel.
        weights (Optional[Sequence[float]]): A list of weights to skew the average of the outputs.
            If provided, must have the same length as `transforms`. If not provided, outputs are equally averaged.
    
    Returns:
        The weighted (skewed) average output of all transforms.
    
    Raises:
        ValueError: If no transforms are provided, or if the length of weights does not match the number of transforms.
        TypeError: If any transform's output is not a torch.Tensor.
    """
    transforms: Sequence[Callable[[Any], Any]]
    weights: Optional[Sequence[float]] = None

    def __call__(self, input_data: Any) -> Any:
        if not self.transforms:
            raise ValueError("No transforms provided to SkewAverages.")
        
        # Apply each transform on the same input.
        outputs = [transform(input_data) for transform in self.transforms]
        
        # Validate that all outputs are torch.Tensors.
        for out in outputs:
            if not isinstance(out, torch.Tensor):
                raise TypeError("All transform outputs must be torch.Tensor")
        
        if self.weights is None:
            # If no weights provided, use equal weighting.
            avg_output = sum(outputs) / len(outputs)
        else:
            if len(self.weights) != len(self.transforms):
                raise ValueError("Length of weights must match the number of transforms.")
            # Convert weights to a tensor (on the same device and dtype as outputs).
            weights_tensor = torch.tensor(self.weights, dtype=outputs[0].dtype, device=outputs[0].device)
            # Compute weighted sum and normalize by the sum of weights.
            weighted_sum = sum(w * out for w, out in zip(weights_tensor, outputs))
            avg_output = weighted_sum / weights_tensor.sum()
        
        return avg_output
