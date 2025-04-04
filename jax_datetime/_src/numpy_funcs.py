# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NumPy-like functions for working with jax_datetime objects."""
from __future__ import annotations

import operator
import typing

import jax
import jax.numpy as jnp
from jax_datetime._src import core
import numpy as np


Array = np.ndarray | jax.Array
ArrayLike = typing.TypeVar(
    'ArrayLike', bound=Array | core.Datetime | core.Timedelta
)


def searchsorted(
    sorted_arr: ArrayLike, query: ArrayLike, side: str = 'left'
) -> jnp.ndarray:
  """jnp.searchsorted() for Datetime and Timedelta arrays."""
  # This implementation here is copied from _searchsorted_via_compare_all() and
  # is only suitable for small arrays:
  # https://github.com/jax-ml/jax/blob/494c15733c5b600f64c3128978c143a82537a830/jax/_src/numpy/lax_numpy.py#L13124
  op = operator.lt if side == 'left' else operator.le
  comparisons = jax.vmap(op, in_axes=(0, None))(sorted_arr, query)
  return comparisons.sum(axis=0, dtype=jnp.int32)


def interp(x: ArrayLike, xp: ArrayLike, fp: Array) -> Array:
  """jnp.interp() for Datetime and Timedelta arrays."""
  # This implementation is adapted with simplifications from
  # https://github.com/jax-ml/jax/blob/494c15733c5b600f64c3128978c143a82537a830/jax/_src/numpy/lax_numpy.py#L2764
  if jnp.shape(xp) != jnp.shape(fp) or jnp.ndim(xp) != 1:
    raise ValueError('xp and fp must be one-dimensional arrays of equal size')
  i = jnp.clip(searchsorted(xp, x, side='right'), 1, len(xp) - 1)
  df = fp[i] - fp[i - 1]
  dx = xp[i] - xp[i - 1]
  delta = x - xp[i - 1]
  f = fp[i - 1] + (delta / dx) * df
  f = jnp.where(x < xp[0], fp[0], f)
  f = jnp.where(x > xp[-1], fp[-1], f)
  return f
