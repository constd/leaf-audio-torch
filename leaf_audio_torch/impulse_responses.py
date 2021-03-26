# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate impulse responses for several type of filters."""

import math
import tensorflow.compat.v2 as tf
import torch as pt


def amplitude(filters: pt.Tensor, i: int) -> float:
    return pt.abs(pt.fft.fftshift(pt.fft.fft(filters[:, :, i])))


def gabor_impulse_response(
    t: pt.Tensor, center: pt.Tensor, fwhm: pt.Tensor
) -> pt.Tensor:
    """Computes the gabor impulse response."""
    denominator = 1.0 / (pt.sqrt(pt.tensor(2.0 * math.pi)) * fwhm)
    gaussian = pt.exp(pt.dot(1.0 / (2.0 * fwhm ** 2), -(t ** 2), dim=0))
    center_frequency_complex = center.type(pt.complex64)
    t_complex = t.type(pt.complex64)
    sinusoid = pt.exp(1j * pt.tensordot(center_frequency_complex, t_complex, dim=0))
    denominator = denominator.type(pt.complex64).unsqueeze(1)
    gaussian = gaussian.type(pt.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401) -> pt.Tensor:
    """Computes the gabor filters from its parameters for a given size.

  Args:
    kernel: tf.Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.

  Returns:
    A tf.Tensor<float>[filters, size].
  """
    return gabor_impulse_response(
        pt.range(-(size // 2), (size + 1) // 2, dtype=pt.float32),
        center=kernel[:, 0],
        fwhm=kernel[:, 1],
    )


def sinc_impulse_response(t: pt.Tensor, frequency: pt.Tensor) -> pt.Tensor:
    """Computes the sinc impulse response."""
    pi = pt.tensor(math.pi)
    return pt.sin(2 * pi * frequency * t) / (2 * pi * frequency * t)


def sinc_filters(
    cutoff_freq_low: pt.Tensor,
    cutoff_freq_high: pt.Tensor,
    size: int = 401,
    sample_rate: int = 16000,
) -> pt.Tensor:
    """Computes the sinc filters from its parameters for a given size.

  Sinc is not defined in zero so we need to separately compute negative
  (left_range) and positive part (right_range).

  Args:
    cutoff_freq_low: tf.Tensor<float>[1, filters] the lower cutoff frequencies
      of the bandpass.
    cutoff_freq_high: tf.Tensor<float>[1, filters] the upper cutoff frequencies
      of the bandpass.
    size: the size of the output tensor.
    sample_rate: audio sampling rate

  Returns:
    A tf.Tensor<float>[size, filters].
  """
    left_range = pt.range(-(size // 2), 0, dtype=pt.float32).unsqueeze(
        -1
    ) / sample_rate.type(pt.float32)
    right_range = pt.range(1, size // 2 + 1, dtype=pt.float32).unsqueeze(
        -1
    ) / sample_rate.type(pt.float32)
    high_pass_left_range = (
        2 * cutoff_freq_high * sinc_impulse_response(left_range, cutoff_freq_high)
    )
    high_pass_right_range = (
        2 * cutoff_freq_high * sinc_impulse_response(right_range, cutoff_freq_high)
    )
    low_pass_left_range = (
        2 * cutoff_freq_low * sinc_impulse_response(left_range, cutoff_freq_low)
    )
    low_pass_right_range = (
        2 * cutoff_freq_low * sinc_impulse_response(right_range, cutoff_freq_low)
    )
    high_pass = pt.cat(
        [high_pass_left_range, 2 * cutoff_freq_high, high_pass_right_range], dim=0
    )
    low_pass = pt.cat(
        [low_pass_left_range, 2 * cutoff_freq_low, low_pass_right_range], dim=0
    )
    band_pass = high_pass - low_pass
    return band_pass / pt.max(band_pass, dim=0, keepdim=True)


def gaussian_lowpass(sigma: tf.Tensor, filter_size: int):
    """Generates gaussian windows centered in zero, of std sigma.

  Args:
    sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.

  Returns:
    A tf.Tensor<float>[1, filter_size, C, 1].
  """
    sigma = pt.clamp(
        sigma, min=(2.0 / filter_size), maxmax=0.5
    )
    t = pt.range(0, filter_size, dtype=pt.float32)
    t = pt.reshape(t, (1, filter_size, 1, 1))
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return pt.exp(-0.5 * (numerator / denominator) ** 2)
