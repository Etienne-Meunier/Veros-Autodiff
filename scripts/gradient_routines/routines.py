"""
Different ways to compute d(loss)/d(param) over an n-step veros rollout.

Every routine has the same signature:
    make_*_grad(step_fn, agg_fn, var_name, iterations) -> jitted (var_value, state) -> (loss, grad)

The only thing that changes is *how the rollout gets traced into XLA*:

- loop:            python for-loop, whole thing jit'd as one XLA program. Traces
                    `iterations` copies of the step, so compile time (and the size of
                    the program XLA has to optimize) grows with rollout length. This is
                    what scripts/report/common.py currently does -- fine for short
                    rollouts, but compile time blows up for long ones.
- loop_checkpoint: same loop, each step wrapped in jax.checkpoint (remat) so the
                    backward pass doesn't keep every intermediate in memory. Still
                    traced `iterations` times -> same compile-time growth as `loop`.
- scan:            jax.lax.scan replaces the python loop -> the step function is traced
                    ONCE regardless of `iterations` (iterations only sets scan length).
                    Compile time stays ~flat as rollout length grows.
- scan_checkpoint: scan + jax.checkpoint on the step -> flat compile time AND bounded
                    backward-pass memory. This is the routine to use for long rollouts
                    (several hundred steps).
"""
import jax
from functools import partial

from scripts.gradient_routines.common import set_var


def _loss_fn(var_value, state, step_fn, agg_fn, var_name, iterations, rollout):
    n_state = set_var(state, var_name, var_value)
    n_state = rollout(step_fn, n_state, iterations)
    return agg_fn(n_state)


def _rollout_loop(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state


def _rollout_scan(step_fn, state, iterations):
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def make_loop_grad(step_fn, agg_fn, var_name, iterations):
    loss_fn = partial(_loss_fn, step_fn=step_fn, agg_fn=agg_fn, var_name=var_name,
                       iterations=iterations, rollout=_rollout_loop)
    return jax.jit(jax.value_and_grad(loss_fn))


def make_loop_checkpoint_grad(step_fn, agg_fn, var_name, iterations):
    return make_loop_grad(jax.checkpoint(step_fn), agg_fn, var_name, iterations)


def make_scan_grad(step_fn, agg_fn, var_name, iterations):
    loss_fn = partial(_loss_fn, step_fn=step_fn, agg_fn=agg_fn, var_name=var_name,
                       iterations=iterations, rollout=_rollout_scan)
    return jax.jit(jax.value_and_grad(loss_fn))


def make_scan_checkpoint_grad(step_fn, agg_fn, var_name, iterations):
    return make_scan_grad(jax.checkpoint(step_fn), agg_fn, var_name, iterations)


ROUTINES = {
    "loop": make_loop_grad,
    "loop_checkpoint": make_loop_checkpoint_grad,
    "scan": make_scan_grad,
    "scan_checkpoint": make_scan_checkpoint_grad,
}
