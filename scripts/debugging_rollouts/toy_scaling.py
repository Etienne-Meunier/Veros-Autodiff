"""Toy reproducer for this directory's rollout strategies -- adapted from
/Users/emeunier/Desktop/Projets/Mini-Autodiff/dynamic_autodiff.py (fixed a few bugs
along the way: an incomplete `inner_scan` def, `jax.checkpoint` wrapping an
already-*executed* scan result instead of a function, a stray bare `r`).

dyn(x, alpha) = sin(x) * alpha -- scalar state, trivial to compile/run. This does
NOT reproduce the real memory wall found on Veros (state there is a big pytree;
here it's one float) -- only useful for cheaply checking whether a strategy's
COMPILE TIME blows up at the n_full scale a very long rollout (e.g. n=10000) would
need, in seconds instead of the minutes-per-config real Veros runs cost. Mirrors
common.py's STRATEGIES exactly (same step_fn(x) -> x signature, same nesting) so a
structure validated fast/cheap here is the same computation graph shape as its Veros
counterpart -- informative about compile-time scaling, not memory.

Run directly: python toy_scaling.py
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def dyn(x, alpha):
    return jnp.sin(x) * alpha


def _run_chunk_scan(step_fn, x, length):
    x, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), x, length=length)
    return x


def _run_chunk_unrolled(step_fn, x, length):
    for _ in range(length):
        x = step_fn(x)
    return x


def rollout_plain(step_fn, x, iterations, chunk_size=None):
    x, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), x, length=iterations)
    return x


def rollout_split_transpose(step_fn, x, iterations, chunk_size=None):
    cp_step = jax.checkpoint(step_fn, policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
    x, _ = jax.lax.scan(lambda c, _: (cp_step(c), None), x, None, length=iterations, _split_transpose=True)
    return x


def rollout_scan_scan(step_fn, x, iterations, chunk_size):
    """The nested-scan design that SIGKILL'd at compile time on Veros (n=10,
    chunk=4)."""
    n_full, remainder = divmod(iterations, chunk_size)
    if n_full:
        ckpt_chunk = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, chunk_size))
        x, _ = jax.lax.scan(lambda c, _: (ckpt_chunk(c), None), x, length=n_full)
    if remainder:
        x = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, remainder))(x)
    return x


def rollout_double_checkpoint(step_fn, x, iterations, chunk_size):
    """The design that actually worked on Veros up to n=1000 (chunk_size=8): per-step
    checkpoint inside the inner scan, plus an outer checkpoint around the block."""
    n_full, remainder = divmod(iterations, chunk_size)
    step_ckpt = jax.checkpoint(step_fn)

    def block_fn(s, length):
        s, _ = jax.lax.scan(lambda c, _: (step_ckpt(c), None), s, length=length)
        return s

    if n_full:
        block_ckpt = jax.checkpoint(lambda s: block_fn(s, chunk_size))
        x, _ = jax.lax.scan(lambda c, _: (block_ckpt(c), None), x, length=n_full)
    if remainder:
        x = jax.checkpoint(lambda s: block_fn(s, remainder))(x)
    return x


def rollout_scan_unrolled(step_fn, x, iterations, chunk_size):
    n_full, remainder = divmod(iterations, chunk_size)
    if n_full:
        ckpt_chunk = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, chunk_size))
        x, _ = jax.lax.scan(lambda c, _: (ckpt_chunk(c), None), x, length=n_full)
    if remainder:
        x = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, remainder))(x)
    return x


def rollout_unrolled_scan(step_fn, x, iterations, chunk_size):
    n_full, remainder = divmod(iterations, chunk_size)
    ckpt_chunk = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, chunk_size)) if n_full else None
    for _ in range(n_full):
        x = ckpt_chunk(x)
    if remainder:
        x = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, remainder))(x)
    return x


def rollout_unrolled_unrolled(step_fn, x, iterations, chunk_size):
    n_full, remainder = divmod(iterations, chunk_size)
    ckpt_chunk = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, chunk_size)) if n_full else None
    for _ in range(n_full):
        x = ckpt_chunk(x)
    if remainder:
        x = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, remainder))(x)
    return x


STRATEGIES = {
    "plain": rollout_plain,
    "split_transpose": rollout_split_transpose,
    "scan_scan": rollout_scan_scan,
    "double_checkpoint": rollout_double_checkpoint,
    "scan_unrolled": rollout_scan_unrolled,
    "unrolled_scan": rollout_unrolled_scan,
    "unrolled_unrolled": rollout_unrolled_unrolled,
}


def time_strategy(strategy, n, chunk_size, alpha_val=1.4, x_init_val=1.1, timeout_hint_s=60):
    rollout_fn = STRATEGIES[strategy]

    def loss(alpha):
        step = lambda x: dyn(x, alpha)
        xn = rollout_fn(step, jnp.array(x_init_val), n, chunk_size)
        return xn ** 2

    grad_fn = jax.grad(loss)
    alpha = jnp.array(alpha_val)

    t0 = time.time()
    try:
        compiled = jax.jit(grad_fn).lower(alpha).compile()
        t1 = time.time()
        g = compiled(alpha)
        jax.block_until_ready(g)
        t2 = time.time()
        return dict(status="OK", compile_time_s=t1 - t0, run_time_s=t2 - t1, grad=float(g))
    except Exception as e:
        t1 = time.time()
        return dict(status=f"ERROR({type(e).__name__}: {str(e)[:200]})", compile_time_s=t1 - t0, run_time_s=None, grad=None)


if __name__ == "__main__":
    N_VALUES = [1000, 10000, 100000]
    CHUNK_SIZES = [8, 32, 128, 512]
    STRATEGIES_TO_TRY = ["scan_scan", "double_checkpoint"]

    for strategy in STRATEGIES_TO_TRY:
        for n in N_VALUES:
            for chunk_size in CHUNK_SIZES:
                if chunk_size > n:
                    continue
                r = time_strategy(strategy, n, chunk_size)
                print(f"[{strategy}][n={n}][chunk={chunk_size}] {r['status']}  "
                      f"compile={r['compile_time_s']:.3f}s  run={r['run_time_s']}  grad={r['grad']}", flush=True)
