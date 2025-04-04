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
import datetime

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np


class TimedeltaTest(parameterized.TestCase):

  def test_constructor(self):
    actual = jdt.Timedelta()
    expected = jdt.Timedelta(0, 0)
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(seconds=jnp.arange(3))
    expected = jdt.Timedelta(days=jnp.zeros(3, int), seconds=jnp.arange(3))
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(days=jnp.arange(3))
    expected = jdt.Timedelta(days=jnp.arange(3), seconds=jnp.zeros(3, int))
    chex.assert_trees_all_equal(actual, expected)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'days and seconds must have the same shape, got () and (3,)',
    ):
      jdt.Timedelta(days=0, seconds=jnp.arange(3))

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'days must be an integer array, got float64'
    ):
      jdt.Timedelta(days=1.5)

  def test_wrapped_array_types(self):
    delta = jdt.Timedelta()
    self.assertIsInstance(delta.days, np.ndarray)
    self.assertIsInstance(delta.seconds, np.ndarray)

    delta = jdt.Timedelta(days=1, seconds=1)
    self.assertIsInstance(delta.days, np.ndarray)
    self.assertIsInstance(delta.seconds, np.ndarray)

    delta = jdt.Timedelta(days=jnp.array(1))
    self.assertIsInstance(delta.days, jnp.ndarray)
    self.assertIsInstance(delta.seconds, jnp.ndarray)

    delta = jdt.Timedelta(seconds=jnp.array(1))
    self.assertIsInstance(delta.days, jnp.ndarray)
    self.assertIsInstance(delta.seconds, jnp.ndarray)

    # TODO(shoyer): consider revising this, to require that days and seconds use
    # the same array type.
    delta = jdt.Timedelta(days=jnp.array(1), seconds=1)
    self.assertIsInstance(delta.days, jnp.ndarray)
    self.assertIsInstance(delta.seconds, np.ndarray)

  def test_array_properties(self):
    delta = jdt.Timedelta(days=jnp.arange(3), seconds=jnp.arange(3))
    self.assertEqual(delta.shape, (3,))
    self.assertEqual(delta.size, 3)
    self.assertLen(delta, 3)
    self.assertEqual(delta.ndim, 1)
    self.assertEqual(delta[1], jdt.Timedelta(days=1, seconds=1))

  def test_transpose(self):
    delta = jdt.Timedelta(days=jnp.array([[1, 2]]), seconds=jnp.array([[3, 4]]))
    expected = jdt.Timedelta(
        days=jnp.array([[1], [2]]), seconds=jnp.array([[3], [4]])
    )
    actual = delta.transpose((1, 0))
    chex.assert_trees_all_equal(actual, expected)

  def test_repr(self):
    delta = jdt.Timedelta(1, 2)
    self.assertEqual(repr(delta), 'jax_datetime.Timedelta(days=1, seconds=2)')

  def test_normalization(self):
    actual = jdt.Timedelta(0, 24 * 60 * 60)
    expected = jdt.Timedelta(1, 0)
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(0, -1)
    expected = jdt.Timedelta(-1, 86399)
    chex.assert_trees_all_equal(actual, expected)

  def test_from_normalized(self):
    expected = jdt.Timedelta(365, 60)
    actual = jdt.Timedelta.from_normalized(365, 60)
    chex.assert_trees_all_equal(actual, expected)

  def test_from_timedelta64(self):
    expected = jdt.Timedelta(365, 0)
    actual = jdt.Timedelta.from_timedelta64(np.timedelta64(365, 'D'))
    chex.assert_trees_all_equal(actual, expected)

    expected = jdt.Timedelta(days=1000 * 365, seconds=10)
    actual = jdt.Timedelta.from_timedelta64(
        np.timedelta64(1000 * 365, 'D') + np.timedelta64(10, 's')
    )
    chex.assert_trees_all_equal(actual, expected)

    expected = jdt.Timedelta(seconds=60 * 60 * np.array([0, 1]))
    actual = jdt.Timedelta.from_timedelta64(
        np.array([0, 1], dtype='timedelta64[h]')
    )
    chex.assert_trees_all_equal(actual, expected)

  def test_from_pytimedelta(self):
    expected = jdt.Timedelta(365, 0)
    actual = jdt.Timedelta.from_pytimedelta(datetime.timedelta(days=365))
    chex.assert_trees_all_equal(actual, expected)

  def test_to_timedelta64(self):
    delta = jdt.Timedelta(365, 0)
    expected = np.timedelta64(365, 'D')
    actual = delta.to_timedelta64()
    self.assertEqual(actual, expected)

  def test_to_pytimedelta(self):
    delta = jdt.Timedelta(365, 0)
    expected = datetime.timedelta(days=365)
    actual = delta.to_pytimedelta()
    self.assertEqual(actual, expected)

    delta = jdt.Timedelta(days=np.int64(365))
    actual = delta.to_pytimedelta()
    self.assertEqual(actual, expected)

  def test_total_seconds(self):
    delta = jdt.Timedelta(days=2, seconds=3)
    expected = 2 * 24 * 60 * 60 + 3
    actual = delta.total_seconds()
    self.assertEqual(actual, expected)
    self.assertEqual(actual.dtype, jnp.float32)

  def test_addition(self):
    delta = jdt.Timedelta(1, 12 * 60 * 60)
    actual = delta + delta
    expected = jdt.Timedelta(3, 0)
    chex.assert_trees_all_equal(actual, expected)

    actual = delta + delta.to_pytimedelta()
    chex.assert_trees_all_equal(actual, expected)

    actual = delta + delta.to_timedelta64()
    chex.assert_trees_all_equal(actual, expected)

    with self.assertRaises(TypeError):
      expected + 60

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'arithmetic between jax_datetime.Timedelta and np.ndarray objects is'
        ' not yet supported. Use jdt.to_datetime() or jdt.to_timedelta() to'
        ' explicitly cast the NumPy array to a Datetime or Timedelta.',
    ):
      delta + np.array(delta.to_timedelta64())

  def test_positive(self):
    delta = jdt.Timedelta(1, 6 * 60 * 60)
    actual = +delta
    chex.assert_trees_all_equal(actual, delta)

  def test_negation(self):
    delta = jdt.Timedelta(1, 6 * 60 * 60)
    actual = -delta
    expected = jdt.Timedelta(-2, 18 * 60 * 60)
    chex.assert_trees_all_equal(actual, expected)

    delta = jdt.Timedelta(-1, 6 * 60 * 60)
    actual = -delta
    expected = jdt.Timedelta(0, 18 * 60 * 60)
    chex.assert_trees_all_equal(actual, expected)

  def test_absolute(self):
    delta = jdt.Timedelta(1, 6 * 60 * 60)
    actual = abs(delta)
    expected = jdt.Timedelta(1, 6 * 60 * 60)
    chex.assert_trees_all_equal(actual, expected)

    delta = jdt.Timedelta(-1, 6 * 60 * 60)
    actual = abs(delta)
    expected = jdt.Timedelta(0, 18 * 60 * 60)
    chex.assert_trees_all_equal(actual, expected)

  def test_subtraction(self):
    delta = jdt.Timedelta(1, 12 * 60 * 60)
    actual = delta - delta
    expected = jdt.Timedelta(0, 0)
    chex.assert_trees_all_equal(actual, expected)

    actual = delta - delta.to_pytimedelta()
    chex.assert_trees_all_equal(actual, expected)

  def test_multiplication_by_integer(self):
    delta = jdt.Timedelta(1, 12 * 60 * 60)
    expected = jdt.Timedelta(3, 0)

    actual = delta * 2
    chex.assert_trees_all_equal(actual, expected)

    actual = 2 * delta
    chex.assert_trees_all_equal(actual, expected)

    actual = delta * True
    chex.assert_trees_all_equal(actual, delta)

  def test_multiplication_by_float(self):
    delta = jdt.Timedelta(1)
    expected = jdt.Timedelta(2, 12 * 60 * 60)
    actual = delta * 2.5
    chex.assert_trees_all_equal(actual, expected)

  def test_multiplication_type_error(self):
    delta = jdt.Timedelta()
    with self.assertRaises(TypeError):
      delta * delta

  def test_division_with_numeric(self):
    actual = jdt.Timedelta(days=2) / 4
    expected = jdt.Timedelta(seconds=12 * 60 * 60)
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(days=4, seconds=-1) / 4
    expected = jdt.Timedelta(days=1)  # rounded to the nearest second
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(days=6, seconds=59) / 4
    expected = jdt.Timedelta(days=1, seconds=12 * 60 * 60 + 15)
    self.assertEqual(actual, expected)

    actual = jdt.Timedelta(days=4, seconds=-1) / 4.0
    expected = jdt.Timedelta(days=1)  # also rounded
    self.assertEqual(actual, expected)

    actual = jdt.Timedelta(days=4, seconds=-1) // 4
    expected = jdt.Timedelta(days=1, seconds=-1)  # rounded down
    chex.assert_trees_all_equal(actual, expected)

    with self.assertRaises(TypeError):
      jdt.Timedelta(days=4) // 4.0  # floor division by a float is not supported

  def test_division_with_timedelta(self):
    actual = jdt.Timedelta(days=2) / jdt.Timedelta(seconds=12 * 60 * 60)
    expected = 4
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(days=3) / jdt.Timedelta(days=2)
    expected = 1.5
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(days=3) // jdt.Timedelta(days=2)
    expected = 1
    chex.assert_trees_all_equal(actual, expected)

    actual = jdt.Timedelta(days=68 * 365) // jdt.Timedelta(days=365)
    expected = 68
    chex.assert_trees_all_equal(actual, expected)

    # needs integer path
    actual = jdt.Timedelta(days=68 * 365, seconds=1) // jdt.Timedelta(seconds=1)
    expected = 68 * 365 * 24 * 60 * 60 + 1
    chex.assert_trees_all_equal(actual, expected)

    # needs float path
    actual = jdt.Timedelta(days=1_000_000 * 365) // jdt.Timedelta(days=365)
    expected = 1_000_000
    chex.assert_trees_all_equal(actual, expected)

  def test_comparison_scalar(self):
    self.assertTrue(jdt.Timedelta(1, 0) == jdt.Timedelta(1, 0))
    self.assertFalse(jdt.Timedelta(1, 0) == jdt.Timedelta(2, 0))
    self.assertTrue(jdt.Timedelta(1, 0) != jdt.Timedelta(2, 0))
    self.assertTrue(jdt.Timedelta(1, 0) < jdt.Timedelta(2, 0))
    self.assertTrue(jdt.Timedelta(0, 0) < jdt.Timedelta(0, 1))
    self.assertFalse(jdt.Timedelta(0, 0) < jdt.Timedelta(0, 0))
    self.assertTrue(jdt.Timedelta(1, 0) <= jdt.Timedelta(1, 0))
    self.assertTrue(jdt.Timedelta(1, 0) > jdt.Timedelta(0, 0))
    self.assertFalse(jdt.Timedelta(0, 0) > jdt.Timedelta(0, 0))
    self.assertTrue(jdt.Timedelta(0, 0) >= jdt.Timedelta(0, 0))

  def test_comparison_invalid(self):
    self.assertFalse(jdt.Timedelta(1, 0) == 1)

    with self.assertRaises(TypeError):
      jdt.Timedelta(1, 0) > 0

  def test_comparison_array(self):
    actual = jdt.Timedelta(days=jnp.arange(3)) < jdt.Timedelta(days=1)
    expected = jnp.array([True, False, False])
    np.testing.assert_array_equal(actual, expected)

    actual = jdt.Timedelta(days=jnp.arange(3)) == jdt.Timedelta(days=1)
    expected = jnp.array([False, True, False])
    np.testing.assert_array_equal(actual, expected)

    actual = jdt.Timedelta(days=jnp.arange(3)) != jdt.Timedelta(days=1)
    expected = jnp.array([True, False, True])
    np.testing.assert_array_equal(actual, expected)

  def test_pytree_transformation_does_not_normalize(self):
    delta = jdt.Timedelta(1, 12 * 60 * 60)
    actual = jax.tree.map(lambda x: 2 * x, delta)
    self.assertEqual(actual.days, 2)
    self.assertEqual(actual.seconds, 24 * 60 * 60)

  def test_vmap(self):
    delta = jdt.Timedelta(days=jnp.arange(2), seconds=jnp.arange(2))
    result = jax.vmap(lambda x: x)(delta)
    self.assertIsInstance(result, jdt.Timedelta)
    np.testing.assert_array_equal(result.days, delta.days)
    np.testing.assert_array_equal(result.seconds, delta.seconds)

  @parameterized.parameters(
      {'value': jdt.Timedelta(days=1, seconds=1)},
      {'value': datetime.timedelta(days=1, seconds=1)},
      {'value': np.timedelta64(24 * 60 * 60 + 1, 's')},
      {'value': np.array(np.timedelta64(24 * 60 * 60 + 1, 's'))},
  )
  def test_to_timedelta(self, value):
    expected = jdt.Timedelta(days=1, seconds=1)
    actual = jdt.to_timedelta(value)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      {'value': 1, 'unit': 'D'},
      {'value': 1, 'unit': 'day'},
      {'value': 1, 'unit': 'days'},
      {'value': 24, 'unit': 'h'},
      {'value': 24, 'unit': 'hr'},
      {'value': 24, 'unit': 'hour'},
      {'value': 24, 'unit': 'hours'},
      {'value': 24 * 60, 'unit': 'm'},
      {'value': 24 * 60, 'unit': 'min'},
      {'value': 24 * 60, 'unit': 'minute'},
      {'value': 24 * 60, 'unit': 'minutes'},
      {'value': 24 * 60 * 60, 'unit': 's'},
      {'value': 24 * 60 * 60, 'unit': 'sec'},
      {'value': 24 * 60 * 60, 'unit': 'second'},
      {'value': 24 * 60 * 60, 'unit': 'seconds'},
  )
  def test_to_timedelta_unit(self, value, unit):
    expected = jdt.Timedelta(days=1)
    actual = jdt.to_timedelta(value, unit)
    chex.assert_trees_all_equal(actual, expected)

  def test_to_timedelta_unit_invalid(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'to_timedelta with units requires either a number or an array of'
        " numbers, got <class 'list'>: [1]",
    ):
      jdt.to_timedelta([1], 'D')

    with self.assertRaisesWithLiteralMatch(
        ValueError, "unsupported unit for to_timedelta: 'weeks'"
    ):
      jdt.to_timedelta(1, 'weeks')


class DatetimeTest(parameterized.TestCase):

  def test_repr(self):
    time = jdt.Datetime(jdt.Timedelta(1))
    self.assertEqual(
        repr(time),
        'jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=1,'
        ' seconds=0))',
    )

  def test_array_properties(self):
    delta = jdt.Timedelta(days=jnp.arange(3), seconds=jnp.arange(3))
    time = jdt.Datetime(delta)
    self.assertEqual(time.shape, (3,))
    self.assertEqual(time.size, 3)
    self.assertLen(time, 3)
    self.assertEqual(time.ndim, 1)
    self.assertEqual(time[1], jdt.Datetime(jdt.Timedelta(days=1, seconds=1)))

  def test_from_datetime64(self):
    expected = jdt.Datetime(jdt.Timedelta(365, 0))
    actual = jdt.Datetime.from_datetime64(np.datetime64('1971-01-01'))
    chex.assert_trees_all_equal(expected, actual)

    expected = jdt.Datetime(jdt.Timedelta(days=np.array([0, 1])))
    actual = jdt.Datetime.from_datetime64(
        np.array(['1970-01-01', '1970-01-02'], dtype='datetime64[D]')
    )
    chex.assert_trees_all_equal(expected, actual)

  def test_from_pydatetime(self):
    expected = jdt.Datetime(jdt.Timedelta(365, 0))
    actual = jdt.Datetime.from_pydatetime(datetime.datetime(1971, 1, 1))
    chex.assert_trees_all_equal(actual, expected)

  def test_from_isoformat(self):
    expected = jdt.Datetime(jdt.Timedelta(365, 0))
    actual = jdt.Datetime.from_isoformat('1971-01-01')
    chex.assert_trees_all_equal(actual, expected)

  def test_to_datetime64(self):
    time = jdt.Datetime(jdt.Timedelta(365, 0))
    expected = np.datetime64('1971-01-01')
    actual = time.to_datetime64()
    self.assertEqual(expected, actual)

  def test_to_pydatetime(self):
    time = jdt.Datetime(jdt.Timedelta(365, 0))
    expected = datetime.datetime(1971, 1, 1)
    actual = time.to_pydatetime()
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      {'original': np.datetime64('2024-01-01T01:02:03', 'ns')},
      {'original': np.datetime64('3024-01-01', 's')},
  )
  def test_datetime64_roundtrip(self, original):
    time = jdt.Datetime.from_datetime64(original)
    restored = time.to_datetime64()
    self.assertEqual(original, restored)

  def test_addition(self):
    delta = jdt.Timedelta(1, 0)
    time = jdt.Datetime(delta)
    expected = jdt.Datetime(delta * 2)

    actual = time + delta
    chex.assert_trees_all_equal(actual, expected)

    actual = time + delta.to_pytimedelta()
    chex.assert_trees_all_equal(actual, expected)

    actual = time + delta.to_timedelta64()
    chex.assert_trees_all_equal(actual, expected)

    actual = time + np.array(delta.to_timedelta64())
    chex.assert_trees_all_equal(actual, expected)

    actual = delta + time
    chex.assert_trees_all_equal(actual, expected)

    actual = delta.to_pytimedelta() + time
    chex.assert_trees_all_equal(actual, expected)

    actual = delta.to_timedelta64() + time
    chex.assert_trees_all_equal(actual, expected)

    actual = np.array(delta.to_timedelta64()) + time
    chex.assert_trees_all_equal(actual, expected)

    with self.assertRaises(TypeError):
      time + 1

    with self.assertRaises(TypeError):
      1 + time

    with self.assertRaises(TypeError):
      time + time

  def test_subtraction(self):
    first = jdt.Datetime.from_isoformat('2024-01-01')
    second = jdt.Datetime.from_isoformat('2023-01-01')
    delta = jdt.Timedelta(365, 0)

    actual = first - delta
    chex.assert_trees_all_equal(actual, second)
    actual = first - delta.to_pytimedelta()
    chex.assert_trees_all_equal(actual, second)
    actual = first - delta.to_timedelta64()
    chex.assert_trees_all_equal(actual, second)

    actual = -delta + first
    chex.assert_trees_all_equal(actual, second)
    actual = -delta.to_pytimedelta() + first
    chex.assert_trees_all_equal(actual, second)
    actual = -delta.to_timedelta64() + first
    chex.assert_trees_all_equal(actual, second)

    actual = first - second
    chex.assert_trees_all_equal(actual, delta)
    actual = first - second.to_pydatetime()
    chex.assert_trees_all_equal(actual, delta)
    actual = first - second.to_datetime64()
    chex.assert_trees_all_equal(actual, delta)

    actual = second - first
    chex.assert_trees_all_equal(actual, -delta)
    actual = second.to_pydatetime() - first
    chex.assert_trees_all_equal(actual, -delta)
    actual = second.to_datetime64() - first
    chex.assert_trees_all_equal(actual, -delta)
    actual = np.array(second.to_datetime64()) - first
    chex.assert_trees_all_equal(actual, -delta)

    with self.assertRaisesWithLiteralMatch(
        TypeError,
        'arithmetic between jax_datetime.Datetime and np.ndarray objects is not'
        ' yet supported. Use jdt.to_datetime() or jdt.to_timedelta() to'
        ' explicitly cast the NumPy array to a Datetime or Timedelta.',
    ):
      first - np.array(second.to_datetime64())

  def test_comparison_scalar(self):
    first = jdt.Datetime.from_isoformat('2020-01-01T00')
    second = jdt.Datetime.from_isoformat('2020-01-01T01')

    self.assertTrue(first == first)
    self.assertFalse(first == second)
    self.assertTrue(first != second)
    self.assertTrue(first < second)
    self.assertFalse(second < first)
    self.assertFalse(first < first)
    self.assertTrue(first <= first)
    self.assertTrue(first <= second)
    self.assertFalse(first > second)
    self.assertTrue(second >= first)

  def test_comparison_invalid(self):
    time = jdt.Datetime.from_isoformat('2024-01-01')
    self.assertFalse(time == 1)

    with self.assertRaises(TypeError):
      time > 0

  def test_comparison_array(self):
    start = jdt.Datetime.from_isoformat('2024-01-01')
    deltas = jdt.Timedelta(days=jnp.arange(3))
    times = start + deltas
    expected = jnp.array([False, True, True])
    actual = times > start
    np.testing.assert_array_equal(actual, expected)

  def test_vmap(self):
    stamp = jdt.Datetime(
        jdt.Timedelta(days=jnp.arange(2), seconds=jnp.arange(2))
    )
    result = jax.vmap(lambda x: x)(stamp)
    self.assertIsInstance(result, jdt.Datetime)
    chex.assert_trees_all_equal(result, stamp)

  @parameterized.parameters(
      {'value': jdt.Datetime.from_isoformat('2000-01-01')},
      {'value': datetime.datetime(2000, 1, 1)},
      {'value': np.datetime64('2000-01-01')},
      {'value': np.array(np.datetime64('2000-01-01'))},
      {'value': '2000-01-01'},
  )
  def test_to_datetime(self, value):
    expected = jdt.Datetime.from_isoformat('2000-01-01')
    actual = jdt.to_datetime(value)
    chex.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
