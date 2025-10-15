import time, pandas as pd
import jax.numpy as jnp
from grad_compare import numerical_diff, jvp_grad, vjp_grad
from utils import warmup_acc

def agg_sum(state, key_sum='u'):
    return ((getattr(state.variables, key_sum)[:, :, :] - 0) ** 2).mean()

def compare_gradients_for_iterations(acc, step_function, agg_function, var_name, var_value, methods, iteration_grad):
    results = []
    for ad_method in methods:
        ad = ad_method(step_function, agg_function, var_name)
        start_time = time.time()
        n_diff = ad.g(acc.state, var_value=var_value, iterations=iteration_grad)
        elapsed = time.time() - start_time
        grad = n_diff[1] if isinstance(n_diff, tuple) else n_diff
        grad_val = float(jnp.mean(grad))
        print(f'{ad} : {grad_val}')
        print(f'Time taken for {ad}: {elapsed:.4f} seconds - its={iteration_grad}\n')
        results.append({
            'method': str(ad),
            'iteration': iteration_grad,
            'gradient': grad_val,
            'time': elapsed
        })
    return results



if __name__ == '__main__':
    acc = warmup_acc(200, override_settings={'enable_streamfunction': False})
    acc.state._diagnostics = {}

    step_function = acc.step
    agg_function = agg_sum
    var_name = 'r_bot'
    var = jnp.array(1e-5, dtype=jnp.float64)

    methods = [numerical_diff, jvp_grad, vjp_grad]
    iteration_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 20, 30, 40, 50]

    all_results = []
    for iteration_grad in iteration_list:
        print(f'--- Iterations: {iteration_grad} ---')
        all_results.extend(compare_gradients_for_iterations(
            acc, step_function, agg_function, var_name, var, methods, iteration_grad
        ))

    df = pd.DataFrame(all_results)
    df.to_csv('gradients_vs_iterations_151025.csv', index=False)