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
"""Core implementation of jax_datetime."""
from __future__ import annotations

import datetime
import functools
import math
import operator
from typing import Self, overload

import jax
import jax.numpy as jnp
import numpy as np


Array = np.ndarray | jax.Array
Integer = int | np.integer | Array
Float = float | np.floating | Integer

DeviceLike = jax.Device | jax.sharding.Sharding | None


class PytreeArray:
  """Base class for an array-like object implemented as a pytree.

  The convention for these objects is that they are built out of a collection
  of JAX arrays with the same shape.

  In principle, they could be implemented as a custom JAX dtype, but such APIs
  are not yet available.

  There is a long list of methods we could potentially implement here. For now,
  we only include the basics.
  """

  @property
  def shape(self) -> tuple[int, ...]:
    shapes = [jnp.shape(x) for x in jax.tree.leaves(self)]
    if len(set(shapes)) != 1:
      raise ValueError('all leaves must have the same shape')
    return shapes[0]

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  def __len__(self) -> int:
    return self.shape[0]

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def transpose(self, axes: tuple[int, ...]) -> PytreeArray:
    return jax.tree.map(lambda x: x.transpose(axes), self)

  def __getitem__(self, index) -> Array:
    return jax.tree.map(lambda x: x[index], self)

  # Take precedence over numpy arrays in binary arithmetic.
  __array_priority__ = 100


def _as_integer_array(x: Integer, name: str) -> np.ndarray | jax.Array:
  if not isinstance(x, jax.Array):
    x = np.asarray(x)
  if not np.issubdtype(x.dtype, np.integer):
    raise ValueError(f'{name} must be an integer array, got {x.dtype}')
  return x


def _zeros_like(other: np.ndarray | jax.Array):
  zeros_like = jnp.zeros_like if isinstance(other, jax.Array) else np.zeros_like
  return zeros_like(other)


_SECONDS_PER_DAY = 24 * 60 * 60


def _asarray(x: np.integer | Array) -> Array:
  return np.asarray(x) if isinstance(x, np.integer) else x


def _normalize_days_seconds(days: Array, seconds: Array) -> tuple[Array, Array]:
  assert np.issubdtype(days.dtype, np.integer)
  assert np.issubdtype(seconds.dtype, np.integer)
  days_delta, seconds = divmod(seconds, _SECONDS_PER_DAY)
  days = _asarray(days + days_delta)
  seconds = _asarray(seconds)
  return (days, seconds)


def _to_int_seconds(delta: Timedelta) -> jnp.ndarray:
  """Returns total seconds as an integer."""
  # This works for timedeltas less than 2**31 seconds, which is ~68 years.
  return delta.days * _SECONDS_PER_DAY + delta.seconds


_INT32_MAX = 2**31 - 1


@jax.jit
def _timedelta_floordiv(
    numerator: Timedelta, divisor: Timedelta
) -> jnp.ndarray:
  """Implements Timedelta // Timedelta as accurately as possible."""
  # We need a separate helper function because it must be jitted to avoid
  # raising OverflowError if int32 calculations are invalid.
  int_calc = _to_int_seconds(numerator) // _to_int_seconds(divisor)
  float_calc = (numerator.total_seconds() // divisor.total_seconds()).astype(
      int
  )
  int32_seconds_valid = (abs(numerator.total_seconds()) < _INT32_MAX) & (
      abs(divisor.total_seconds()) < _INT32_MAX
  )
  return jnp.where(int32_seconds_valid, int_calc, float_calc)


@jax.tree_util.register_pytree_node_class
class Timedelta(PytreeArray):
  """JAX compatible time duration, stored in days and seconds.

  Like datetime.timedelta, the `Timedelta` constructor normalizes the seconds
  field to fall in the range `[0, 24*60*60)`, with whole days moved into `days`.

  Timedelta is implemented internally as a JAX pytree of integer arrays of
  days and seconds. Using JAX's default int32 precision, Timedelta can exactly
  represent durations over 5 million years.

  You can either use the `Timedelta` constructor directly, or use `to_timedelta`
  to convert from `datetime.timedelta`, `np.timedelta64` or integers with
  units::

    >>> import jax_datetime as jdt
    >>> import datetime
    >>> jdt.Timedelta(days=1)
    jax_datetime.Timedelta(days=1, seconds=0)
    >>> jdt.to_timedelta(datetime.timedelta(days=1))
    jax_datetime.Timedelta(days=1, seconds=0)
    >>> jdt.to_timedelta(1, 'D')
    jax_datetime.Timedelta(days=1, seconds=0)

  Attributes:
    days: integer JAX array indicating the number of days in the duration.
    seconds: integer JAX array indicating the number of seconds in the duration,
      normalized to fall in the range `[0, 24*60*60)`.
  """

  # TODO(shoyer): can we rewrite this a custom JAX dtype, like jax.random.key?

  def __init__(
      self, days: Integer | None = None, seconds: Integer | None = None
  ):
    """Construct a Timedelta object.

    Args:
      days: optional number of days in the duration provided as an int or int
        array, defaulting to zero.
      seconds: optional number of seconds in the duration provided as an int or
        int array, defaulting to zero. If both seconds and days are provided,
        they must have the same shape.
    """
    if days is None and seconds is None:
      days = np.asarray(0)
      seconds = np.asarray(0)
    elif days is None:
      seconds = _as_integer_array(seconds, name='seconds')
      days = _zeros_like(seconds)
    elif seconds is None:
      days = _as_integer_array(days, name='days')
      seconds = _zeros_like(days)
    else:
      days = _as_integer_array(days, name='days')
      seconds = _as_integer_array(seconds, name='seconds')
      if days.shape != seconds.shape:
        raise ValueError(
            f'days and seconds must have the same shape, got {days.shape} and'
            f' {seconds.shape}'
        )

    self._days, self._seconds = _normalize_days_seconds(days, seconds)

  @property
  def days(self) -> Array:
    return self._days

  @property
  def seconds(self) -> Array:
    return self._seconds

  def __repr__(self) -> str:
    return f'jax_datetime.Timedelta(days={self.days}, seconds={self.seconds})'

  @classmethod
  def from_normalized(cls, days: Integer, seconds: Integer) -> Self:
    """Fast-path constructor from pre-normalized days and seconds."""
    result = object.__new__(cls)
    result._days = _as_integer_array(days, name='days')
    result._seconds = _as_integer_array(seconds, name='seconds')
    return result

  @classmethod
  def from_timedelta64(cls, values: np.timedelta64 | np.ndarray) -> Self:
    """Construct a Timedelta from a NumPy timedelta64 scalar or array."""
    seconds = values // np.timedelta64(1, 's')  # round down
    # normalize with numpy int64 arrays to avoid overflow in int32
    days, seconds = divmod(seconds, _SECONDS_PER_DAY)
    return cls.from_normalized(days, seconds)

  @classmethod
  def from_pytimedelta(cls, values: datetime.timedelta) -> Self:
    """Construct a Timedelta from a datetime.timedelta object."""
    return cls.from_normalized(values.days, values.seconds)

  def to_timedelta64(self) -> np.timedelta64 | np.ndarray:
    """Convert this value to a np.timedelta64 scalar or array."""
    seconds = np.int64(self.days) * _SECONDS_PER_DAY + np.int64(self.seconds)
    return seconds.astype(dtype='timedelta64[s]')

  def to_pytimedelta(self) -> datetime.timedelta:
    """Convert this value to a datetime.timedelta object."""
    return datetime.timedelta(
        days=operator.index(self.days), seconds=operator.index(self.seconds)
    )

  # The implementation of all methods should match datetime.timedelta, except
  # extended to handle jax.Array objects:
  # https://docs.python.org/3/library/datetime.html#timedelta-objects

  def total_seconds(self) -> jnp.ndarray:
    """Total number of seconds in the duration, as a JAX array of floats."""
    return jnp.asarray(self.days, float) * _SECONDS_PER_DAY + self.seconds

  @overload
  def __add__(self, other: DatetimeLike) -> Datetime:
    ...

  @overload
  def __add__(self, other: TimedeltaLike) -> Timedelta:
    ...

  def __add__(
      self, other: TimedeltaLike | DatetimeLike
  ) -> Timedelta | Datetime:
    if isinstance(other, DatetimeLike):
      other = to_datetime(other)
      return other + self
    elif isinstance(other, TimedeltaLike):
      other = to_timedelta(other)
      days = self.days + other.days
      seconds = self.seconds + other.seconds
      return Timedelta(days, seconds)  # type: ignore
    elif isinstance(other, np.ndarray):
      # TODO(shoyer): consider handling np.ndarray objects. This is tricky to
      # type check because the correct return type depends on the array dtype.
      raise TypeError(
          'arithmetic between jax_datetime.Timedelta and np.ndarray objects is'
          ' not yet supported. Use jdt.to_datetime() or jdt.to_timedelta() to'
          ' explicitly cast the NumPy array to a Datetime or Timedelta.'
      )
    else:
      return NotImplemented  # type: ignore

  __radd__ = __add__

  def __pos__(self) -> Timedelta:
    return self

  def __neg__(self) -> Timedelta:
    return Timedelta(-self.days, -self.seconds)

  def __abs__(self) -> Timedelta:
    return jax.tree.map(
        functools.partial(jnp.where, self.days < 0), -self, self
    )

  def __sub__(self, other: TimedeltaLike) -> Timedelta:
    # TODO(shoyer): consider handling timedelta64 np.ndarray objects
    if not isinstance(other, TimedeltaLike):
      return NotImplemented  # type: ignore
    other = to_timedelta(other)
    return self + (-other)  # type: ignore

  def __mul__(self, other: Float | bool) -> Timedelta:
    if not isinstance(other, Float | bool):
      return NotImplemented
    other = jnp.asarray(other)
    if jnp.issubdtype(other.dtype, jnp.integer) or jnp.issubdtype(
        other.dtype, jnp.bool
    ):
      return Timedelta(self.days * other, self.seconds * other)
    elif jnp.issubdtype(other.dtype, jnp.floating):
      float_days, day_fraction = jnp.divmod(self.days * other, 1)
      float_seconds = day_fraction * _SECONDS_PER_DAY + self.seconds * other
      days = jnp.around(float_days).astype(int)
      seconds = jnp.around(float_seconds).astype(int)
      return Timedelta(days, seconds)
    else:
      return NotImplemented  # type: ignore

  __rmul__ = __mul__

  @overload
  def __truediv__(self, other: TimedeltaLike) -> jnp.ndarray:
    ...

  @overload
  def __truediv__(self, other: Float) -> Timedelta:
    ...

  def __truediv__(
      self, other: TimedeltaLike | Float
  ) -> jnp.ndarray | Timedelta:
    if isinstance(other, TimedeltaLike):
      other = to_timedelta(other)
      return self.total_seconds() / other.total_seconds()  # type: ignore
    elif isinstance(other, Float):
      other = jnp.asarray(other)
      if jnp.issubdtype(other.dtype, jnp.integer):
        days, remaining_days = jnp.divmod(self.days, other)
        float_seconds = (
            remaining_days * _SECONDS_PER_DAY + self.seconds
        ) / other
        seconds = jnp.around(float_seconds).astype(int)
        return Timedelta(days, seconds)  # type: ignore
      elif jnp.issubdtype(other.dtype, jnp.floating):
        float_days, remaining_days = jnp.divmod(self.days, other)
        float_seconds = (
            remaining_days * _SECONDS_PER_DAY + self.seconds
        ) / other
        days = jnp.around(float_days).astype(int)
        seconds = jnp.around(float_seconds).astype(int)
        return Timedelta(days, seconds)  # type: ignore
      else:
        return NotImplemented  # type: ignore
    else:
      return NotImplemented  # type: ignore

  @overload
  def __floordiv__(self, other: TimedeltaLike) -> jnp.ndarray:
    ...

  @overload
  def __floordiv__(self, other: Float) -> Timedelta:
    ...

  def __floordiv__(
      self, other: TimedeltaLike | Float
  ) -> jnp.ndarray | Timedelta:
    if isinstance(other, TimedeltaLike):
      other = to_timedelta(other)
      return _timedelta_floordiv(self, other)
    elif isinstance(other, Float):
      other = jnp.asarray(other)
      if not jnp.issubdtype(other.dtype, jnp.integer):
        return NotImplemented  # type: ignore
      days, remaining_days = jnp.divmod(self.days, other)
      seconds = (remaining_days * _SECONDS_PER_DAY + self.seconds) // other
      return Timedelta(days, seconds)  # type: ignore
    else:
      return NotImplemented  # type: ignore

  # TODO(shoyer): implement __divmod__ and __mod__

  def _comparison_op(wrapped):  # pylint: disable=no-self-argument
    """Private decorator for implementing comparison ops."""

    # Disable type errors for mismatched signatures with the base class
    # comparison method (object), which always returns bool.
    def wrapper(self, other: TimedeltaLike) -> jnp.ndarray:  # type: ignore
      if not isinstance(other, TimedeltaLike):
        return NotImplemented  # type: ignore
      other = to_timedelta(other)
      return wrapped(self, other)

    return wrapper

  @_comparison_op
  def __eq__(self, other: Timedelta) -> jnp.ndarray:
    return (self.days == other.days) & (self.seconds == other.seconds)

  @_comparison_op
  def __ne__(self, other: Timedelta) -> jnp.ndarray:
    return (self.days != other.days) | (self.seconds != other.seconds)

  @_comparison_op
  def __lt__(self, other: Timedelta) -> jnp.ndarray:
    return (self.days < other.days) | (
        (self.days == other.days) & (self.seconds < other.seconds)
    )

  @_comparison_op
  def __le__(self, other: Timedelta) -> jnp.ndarray:
    return (self.days < other.days) | (
        (self.days == other.days) & (self.seconds <= other.seconds)
    )

  @_comparison_op
  def __gt__(self, other: Timedelta) -> jnp.ndarray:
    return (self.days > other.days) | (
        (self.days == other.days) & (self.seconds > other.seconds)
    )

  @_comparison_op
  def __ge__(self, other: Timedelta) -> jnp.ndarray:
    return (self.days > other.days) | (
        (self.days == other.days) & (self.seconds >= other.seconds)
    )

  def tree_flatten(self):
    """Custom flatten method for pytree serialization."""
    leaves = (self.days, self.seconds)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    """Custom unflatten method for pytree serialization."""
    assert aux_data is None
    # JAX uses non-numeric values for pytree leaves inside transformations, so
    # we skip __init__ by constructing the object directly:
    # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    result = object.__new__(cls)
    result._days, result._seconds = leaves
    return result


_NUMPY_UNIX_EPOCH = np.datetime64('1970-01-01T00:00:00', 's')
_PY_UNIX_EPOCH = datetime.datetime(1970, 1, 1)


@jax.tree_util.register_pytree_node_class
class Datetime(PytreeArray):
  """JAX compatible datetime, stored as a delta from the Unix epoch.

  The easiest way to create a Datetime is to use `to_datetime`, which
  supports `datetime.datetime`, `np.datetime64` and strings in ISO 8601 format:

    >>> import jax_datetime as jdt
    >>> jdt.to_datetime('1970-01-02')
    jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=1, seconds=0))

  Attributes:
    delta: difference between this date and the Unix epoch (1970-01-01).
  """

  def __init__(self, delta: Timedelta):
    self._delta = delta

  @property
  def delta(self) -> Timedelta:
    return self._delta

  def __repr__(self) -> str:
    return f'jax_datetime.Datetime(delta={self.delta})'

  @classmethod
  def from_datetime64(cls, values: np.datetime64 | np.ndarray) -> Datetime:
    """Construct a Datetime from a NumPy datetime64 scalar or array."""
    return cls(Timedelta.from_timedelta64(values - _NUMPY_UNIX_EPOCH))

  @classmethod
  def from_pydatetime(cls, value: datetime.datetime) -> Datetime:
    """Construct a Datetime from a Python datetime.datetime object."""
    return cls(Timedelta.from_pytimedelta(value - _PY_UNIX_EPOCH))

  @classmethod
  def from_isoformat(cls, value: str) -> Datetime:
    """Construct a Datetime from an ISO 8601 string, e.g., '2024-01-01T00'."""
    return cls.from_pydatetime(datetime.datetime.fromisoformat(value))

  def to_datetime64(self) -> np.datetime64 | np.ndarray:
    """Convert this Datetime to a NumPy datetime64 scalar or array."""
    return self.delta.to_timedelta64() + _NUMPY_UNIX_EPOCH

  def to_pydatetime(self) -> datetime.datetime:
    """Convert this Datetime to a Python datetime.datetime object."""
    return self.delta.to_pytimedelta() + _PY_UNIX_EPOCH

  def __add__(self, other: TimedeltaLike | np.ndarray) -> Datetime:
    if not isinstance(other, TimedeltaLike | np.ndarray):
      return NotImplemented  # type: ignore
    other = to_timedelta(other)
    return Datetime(self.delta + other)  # type: ignore

  __radd__ = __add__

  @overload
  def __sub__(self, other: DatetimeLike) -> Timedelta:
    ...

  @overload
  def __sub__(self, other: TimedeltaLike) -> Datetime:
    ...

  def __sub__(
      self, other: TimedeltaLike | DatetimeLike
  ) -> Timedelta | Datetime:
    if isinstance(other, DatetimeLike):
      other = to_datetime(other)
      return self.delta - other.delta
    elif isinstance(other, TimedeltaLike):
      other = to_timedelta(other)
      return Datetime(self.delta - other)  # type: ignore
    elif isinstance(other, np.ndarray):
      # TODO(shoyer): consider handling np.ndarray objects. This is tricky to
      # type check because the correct return type depends on the array dtype.
      raise TypeError(
          'arithmetic between jax_datetime.Datetime and np.ndarray objects is'
          ' not yet supported. Use jdt.to_datetime() or jdt.to_timedelta() to'
          ' explicitly cast the NumPy array to a Datetime or Timedelta.'
      )
    else:
      return NotImplemented  # type: ignore

  def __rsub__(self, other: DatetimeLike | np.ndarray) -> Timedelta:
    # TODO(shoyer): consider handling datetime64 np.ndarray objects
    if isinstance(other, DatetimeLike | np.ndarray):
      other = to_datetime(other)
      return other.delta - self.delta
    else:
      return NotImplemented  # type: ignore

  def _comparison_op(wrapped):  # pylint: disable=no-self-argument
    """Private decorator for implementing comparison ops."""

    # Disable type errors for mismatched signatures with the base class
    # comparison method (object), which always returns bool.
    def wrapper(self, other: DatetimeLike) -> jnp.ndarray:  # type: ignore
      if not isinstance(other, DatetimeLike):
        return NotImplemented  # type: ignore
      other = to_datetime(other)
      return wrapped(self, other)

    return wrapper

  @_comparison_op
  def __eq__(self, other: Datetime) -> jnp.ndarray:
    return self.delta == other.delta

  @_comparison_op
  def __ne__(self, other: Datetime) -> jnp.ndarray:
    return self.delta != other.delta

  @_comparison_op
  def __lt__(self, other: Datetime) -> jnp.ndarray:
    return self.delta < other.delta

  @_comparison_op
  def __le__(self, other: Datetime) -> jnp.ndarray:
    return self.delta <= other.delta

  @_comparison_op
  def __gt__(self, other: Datetime) -> jnp.ndarray:
    return self.delta > other.delta

  @_comparison_op
  def __ge__(self, other: Datetime) -> jnp.ndarray:
    return self.delta >= other.delta

  def tree_flatten(self):
    leaves = (self.delta,)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    return cls(*leaves)


DatetimeLike = Datetime | datetime.datetime | np.datetime64
TimedeltaLike = Timedelta | datetime.timedelta | np.timedelta64


def to_datetime(value: DatetimeLike | np.ndarray | str) -> Datetime:
  """Convert a value into a Datetime object.

  Args:
    value: a jax_datetime.Datetime, datetime.datetime, np.datetime64, np.ndarray
      with a datetime64 dtype or string in ISO 8601 format.

  Returns:
    Value cast to a jax_datetime.Datetime object.
  """
  match value:
    case Datetime():
      return value
    case datetime.datetime():
      return Datetime.from_pydatetime(value)
    case np.datetime64():
      return Datetime.from_datetime64(value)
    case np.ndarray():
      return Datetime.from_datetime64(value)
    case str():
      return Datetime.from_isoformat(value)
    case _:
      raise TypeError(f'unsupported type for to_datetime: {type(value)}')


def _to_timedelta_from_units(value: Integer, unit: str) -> Timedelta:
  """Create Timedelta from a numeric value and unit string."""
  # valid units are a subset of those supported by pd.to_timedelta:
  # https://pandas.pydata.org/docs/reference/api/pandas.to_timedelta.html
  if not isinstance(value, Integer):
    raise TypeError(
        'to_timedelta with units requires either a number or an array of'
        f' numbers, got {type(value)}: {value!r}'
    )
  if unit in {'D', 'day', 'days'}:
    return Timedelta(days=value)
  elif unit in {'h', 'hr', 'hour', 'hours'}:
    return Timedelta(seconds=value * 3600)
  elif unit in {'m', 'min', 'minute', 'minutes'}:
    return Timedelta(seconds=value * 60)
  elif unit in {'s', 'sec', 'second', 'seconds'}:
    return Timedelta(seconds=value)
  else:
    raise ValueError(f'unsupported unit for to_timedelta: {unit!r}')


def to_timedelta(
    value: TimedeltaLike | Integer,
    unit: str | None = None,
) -> Timedelta:
  """Convert a value into a Timedelta object.

  Args:
    value: a jax_datetime.Timedelta, datetime.timedelta, np.timedelta64,
      np.ndarray with a timedelta64 dtype, array with a numeric dtype or number.
    unit: optional units string. Required if `value` is given as a number.
      Supported values are D/days/days, hours/hour/hr/h/H, m/minute/min/minutes,
      and s/seconds/sec/second, i.e., NumPy's supported datetime units plus
      standard abbreviations.

  Returns:
    Value cast to a jax_datetime.Timedelta object.
  """
  if unit is not None:
    return _to_timedelta_from_units(value, unit)

  match value:
    case Timedelta():
      return value
    case datetime.timedelta():
      return Timedelta.from_pytimedelta(value)
    case np.timedelta64():
      return Timedelta.from_timedelta64(value)
    case np.ndarray():
      return Timedelta.from_timedelta64(value)
    case _:
      raise TypeError(
          f'unsupported type for to_timedelta without unit: {type(value)}'
      )
