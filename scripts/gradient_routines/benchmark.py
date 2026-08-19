# %%
# Compile time + steady-state runtime of each gradient routine (routines.py) vs rollout
# length, on the ACC toy setup (small grid -> can afford several hundred steps).
#
# `loop` / `loop_checkpoint` unroll a python for-loop into the XLA trace, so compile
# time grows with n -- kept to short rollouts here, since a few hundred unrolled steps
# can take a very long time (sometimes minutes) to compile.
# `scan` / `scan_checkpoint` trace the step once via jax.lax.scan, so compile time
# should stay ~flat even at n=hundreds -- that's the routine to use for long rollouts.
from __init__ import PRP
import sys
sys.path.append(PRP)

import time
import jax
import jax.numpy as jnp

from scripts.gradient_routines.common import spin_up_acc, pure_step, agg_sum_sq
from scripts.gradient_routines.routines import ROUTINES

TEST_VAR = jnp.array(2e-5)  # r_bot, off the spun-up true value (1e-5)
VAR_NAME = "r_bot"
N_REPEATS = 3

# loop / loop_checkpoint: compile time grows with n -> keep short.
# scan / scan_checkpoint: should stay flat -> push into the hundreds.
N_VALUES = {
    # n=20 measured separately at several minutes of compile time for `loop` (see
    # README) -- capped much lower here so the sweep finishes in a reasonable time.
    "loop": [2, 5, 10],
    "loop_checkpoint": [2, 5, 10],
    "scan": [2, 5, 10, 20, 40, 100, 200, 400],
    "scan_checkpoint": [2, 5, 10, 20, 40, 100, 200, 400],
}


def time_call(fn, *args, n_repeats=N_REPEATS):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - t0

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return compile_time, min(times)


# %%
if __name__ == "__main__":
    acc = spin_up_acc(200)
    step_fn = pure_step(acc)
    state = acc.state

    out_csv = f"{PRP}Results/gradient_routines_benchmark.csv"
    csv_f = open(out_csv, "w")
    csv_f.write("method,n,compile_time,run_time,loss,grad\n")
    csv_f.flush()

    results = []
    reference = {}
    for name, make_fn in ROUTINES.items():
        for n in N_VALUES[name]:
            grad_fn = make_fn(step_fn, agg_sum_sq, VAR_NAME, n)
            compile_t, run_t = time_call(grad_fn, TEST_VAR, state)
            loss, grad = grad_fn(TEST_VAR, state)
            loss, grad = float(loss), float(grad)

            print(f"{name:16s} n={n:4d}  compile={compile_t:7.3f}s  run={run_t:7.4f}s  "
                  f"loss={loss:.6e}  grad={grad:.6e}", flush=True)
            results.append(dict(method=name, n=n, compile_time=compile_t, run_time=run_t,
                                 loss=loss, grad=grad))
            reference.setdefault(n, {})[name] = (loss, grad)
            csv_f.write(f"{name},{n},{compile_t},{run_t},{loss},{grad}\n")
            csv_f.flush()

    csv_f.close()

    # %%
    print("\ncorrectness check -- loss/grad should match (up to float noise) across "
          "methods sharing the same n:")
    for n, vals in sorted(reference.items()):
        if len(vals) < 2:
            continue
        print(f" n={n}")
        for name, (loss, grad) in vals.items():
            print(f"   {name:16s} loss={loss:.8e} grad={grad:.8e}")

    print(f"\nSaved results to {out_csv}")
