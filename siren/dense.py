# Copyright 2022 The Flax Authors.
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

"""Linear modules."""

import dataclasses
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

from flax.core import meta
from flax.linen import initializers
from flax.linen.module import compact
from flax.linen.module import Module
from flax.linen.dtypes import promote_dtype
from jax import eval_shape
from jax import lax
from jax import random
from jax import ShapedArray
import jax.numpy as jnp
import numpy as np
import jax
from jax import core
from jax._src.nn.initializers import _compute_fans
from jax._src import dtypes


PRNGKey = Any
KeyArray = jax.random.KeyArray
DTypeLikeInexact = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = initializers.lecun_normal()

# init are from https://github.com/KeunwooPark/siren-jax/blob/main/siren/initializer.py
def siren_init_first(in_axis: Union[int, Sequence[int]] = -2,
                   out_axis: Union[int, Sequence[int]] = -1,
                   batch_axis: Sequence[int] = (),
                   dtype: DTypeLikeInexact = jnp.float_):
    def init(key: KeyArray,
        shape: core.Shape,
        dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        fan_in, _ = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
        return jax.random.uniform(key, shape, dtype, minval=-1 / fan_in, maxval=1/fan_in)
    return init

def siren_init(in_axis: Union[int, Sequence[int]] = -2,
               out_axis: Union[int, Sequence[int]] = -1,
               batch_axis: Sequence[int] = (),
               dtype: DTypeLikeInexact = jnp.float_,
               w0=30.0):
    def init(key: KeyArray,
        shape: core.Shape,
        dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        fan_in, _ = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
        return jax.random.uniform(key, shape, dtype, minval=-jnp.sqrt(6 / fan_in) / w0, maxval=jnp.sqrt(6 / fan_in) / w0)
    return init

def bias_uniform(in_axis: Union[int, Sequence[int]] = -2,
               out_axis: Union[int, Sequence[int]] = -1,
               batch_axis: Sequence[int] = (),
               dtype: DTypeLikeInexact = jnp.float_):
    # this is what Pytorch default Linear uses.
    def init(key: KeyArray,
        shape: core.Shape,
        dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
        variance = jnp.sqrt(1 / fan_in)
        return jax.random.uniform( key, (int(fan_out),), dtype, minval=-variance, maxval=variance)

    return init

class CustomDense(Module):
  """A linear transformation applied over the last dimension of the input.

  Modified to allow bias_init to take in inputs (jnp.shape(inputs)[-1], self.features)

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  dot_general: Any = lax.dot_general

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param('kernel',
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    y = self.dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y