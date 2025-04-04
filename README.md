# JAX Datetime: JAX compatible datetime and timedelta types

JAX Datetime implements basic datetime and timedelta functionality in a JAX
compatible fashion. JAX Datetime's `Datetime` and `Timedelta` classes can hold
arrays of values and are JAX pytrees, which makes them compatible with JAX
transformations such as `jax.vmap` and `jax.jit`.

## Typical usage

You can create `Timedelta` and `Datetime` objects either directly, or via the
`to_timedelta` and `to_datetime` helpers, which also handle NumPy and datetime
objects:

    >>> import jax_datetime as jdt

    >>> delta = jdt.to_timedelta(1, 'day')

    >>> delta
    jax_datetime.Timedelta(days=1, seconds=0)

    >>> time = jdt.to_datetime('2000-01-01')

    >>> time
    jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=10957, seconds=0))

`Timedelta` and `Datetime` objects support arithmetic like standard datetime
objects, including with built-in datetime and scalar NumPy objects:

    >>> time + delta
    jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=10958, seconds=0))

    >>> time + datetime.timedelta(days=1)
    jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=10958, seconds=0))

You can also construct them from multidimensional arrays, in which case they
support basic array properties like `shape` and `__getitem__` :

    >>> days = jdt.to_timedelta(jnp.arange(5), 'days')

    >>> days
    jax_datetime.Timedelta(days=[0 1 2 3 4], seconds=[0 0 0 0 0])

    >>> days.shape
    (5,)

    >>> days[-1]
    jax_datetime.Timedelta(days=4, seconds=0)

Finally, you can convert back to standard NumPy or Python datetime objects using
the `to_datetime64`, `to_pydatetime`, `to_timedelta64` and `to_pytimedelta`
methods:

    >>> time.to_pydatetime()
    datetime.datetime(2000, 1, 1, 0, 0)

    >>> delta.to_timedelta64()
    numpy.timedelta64(86400,'s')

## Pytree operations

`Timedelta` and `Datetime` objects are JAX pytrees, which means they can be
used as inputs to JAX transformations such as `jax.vmap`, `jax.jit` and
`jax.lax.scan` (`jax.grad` is not supported, because JAX Datetime uses integers
internally to store data):

    >>> jax.jit(lambda x: x + delta)(time)
    jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=10958, seconds=0))

This is also helpful for re-arranging multi-dimensional arrays of `Timedelta`
and `Datetime` objects, e.g., using `jnp.stack` and `jnp.concat`:

    >>> import jax

    >>> import jax.numpy as jnp

    >>> jax.tree.map(lambda *xs: jnp.stack(xs), time, time + delta)
    jax_datetime.Datetime(delta=jax_datetime.Timedelta(days=[10957 10958], seconds=[0 0]))

In fact, `__getitem__` on `Timedelta` and `Datetime` objects is implemented
in exactly such as a fashion.

**Warning**: Do not modify _values_ on the arrays underlying JAX Datetime
objects directly using JAX pytree operations (e.g., `jax.tree.map`). In such
cases, normalization from JAX Datetime constructors will be skipped, and you may
create invalid objects, for which some operations (e.g., comparisons for
equality) will give silently incorrect results:

    >>> import jax

    >>> hour = jdt.to_timedelta(1, 'hour')

    >>> invalid_delta = jax.tree.map(lambda x: 24 * x, hour)  # don't do this!

    >>> invalid_delta
    jax_datetime.Timedelta(days=0, seconds=86400)

    >>> delta == invalid_delta  # untrue!
    False

## Implementation

Under the hood, `Timedelta` stores its state in integer arrays of `days` and
`seconds`. `Datetime` is implemented as a simple wrapper around `Timedelta`,
indicating a time difference relative to the start of the Unix Epoch
(1970-01-01).

Like datetime.timedelta, the seconds field in `Timedelta` is always normalized
to fall in the range `[0, 24*60*60)`, with whole days moved into `days`. Using
JAX's default int32 precision, Timedelta can exactly represent durations over 5
million years.

Currently, `Timedelta` and `Datetime` objects are implemented as JAX pytrees,
We will likely switch the implementation to make use of custom dtypes if they
are supported by JAX in the future.

The underlying integer array types wrapped by JAX-Datetime may be either NumPy
or JAX arrays. NumPy arrays are preserved by the constructor, but the results of
any computation will likely be JAX arrays.
