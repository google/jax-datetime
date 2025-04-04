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
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np


class NumpyFuncsTest(parameterized.TestCase):

  def test_searchsorted(self):
    sorted_arr = jdt.Timedelta(days=jnp.arange(3))
    query = jdt.Timedelta(days=1)
    expected = np.searchsorted(np.arange(3), 1)
    actual = jdt.searchsorted(sorted_arr, query)
    self.assertEqual(actual, expected)

    sorted_arr = jdt.Timedelta(days=jnp.arange(3), seconds=jnp.arange(3))
    query = jdt.Timedelta(days=2)
    expected = np.searchsorted(np.array([0, 1.1, 2.2]), 2)
    actual = jdt.searchsorted(sorted_arr, query)
    self.assertEqual(actual, expected)

    sorted_arr = jdt.Timedelta(days=jnp.arange(3))
    query = jdt.Timedelta(days=1, seconds=2)
    expected = np.searchsorted(np.array([0, 1, 2]), 1.2)
    actual = jdt.searchsorted(sorted_arr, query)
    self.assertEqual(actual, expected)

    sorted_arr = jdt.Datetime(sorted_arr)
    query = jdt.Datetime(query)
    actual = jdt.searchsorted(sorted_arr, query)
    self.assertEqual(actual, expected)

  def test_interp(self):
    deltas = jdt.Timedelta(days=jnp.arange(3))
    query = jdt.Timedelta(days=1, seconds=0)
    expected = 1.0
    actual = jdt.interp(query, deltas, jnp.arange(3))
    self.assertEqual(actual, expected)

    deltas = jdt.Timedelta(days=jnp.arange(3))
    query = jdt.Timedelta(days=1, seconds=6 * 60 * 60)
    expected = 1.25
    actual = jdt.interp(query, deltas, jnp.arange(3))
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
