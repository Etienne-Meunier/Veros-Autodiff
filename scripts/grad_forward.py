def warmup_acc(steps_bf_check = 300):
    """
    Initialize and warm up the ACC (Antarctic Circumpolar Current) simulation

    Args:
        steps_bf_check (int): Number of simulation steps for warm-up

    Returns:
        acc: Initialized ACC simulation object
    """
    acc = ACCSetup()
    acc.setup()
    # Add small perturbation to bottom friction
    with acc.state.variables.unlock():
        acc.state.variables.r_bot += 1e-5
    # Enable TKE and neutral diffusion settings
    with acc.state.settings.unlock():
        acc.state.settings.enable_tke = True
        acc.state.settings.enable_neutral_diffusion = True
    # Run warm-up steps
    for step in tqdm(range(steps_bf_check)):
        acc.step(acc.state)
    return acc

class autodiff:
    """Base class for automatic differentiation methods"""

    def __init__(self, step_function, agg_function, var_name):
        """
        Initialize autodiff object

        Args:
            step_function: Function to be executed iteratively
            agg_function: Aggregation function to compute final scalar output
            var_name: Name of variable to differentiate with respect to
        """
        self.agg_function = agg_function
        self.step_function = partial(autodiff.pure, step=step_function)
        self.var_name = var_name

    @staticmethod
    def pure(state, step):
        """Convert state-modifying function to pure function"""
        n_state = state.copy()
        step(n_state)
        return n_state

    @staticmethod
    def set_var(var_name, state, var_value):
        """Set variable value in state copy"""
        n_state = state.copy()
        vs = n_state.variables
        with n_state.variables.unlock():
            setattr(vs, var_name, var_value)
        return n_state

class numerical_diff(autodiff):
    """Numerical differentiation implementation"""

    def g(self, state, var_value, iterations=1, eps=1e-9, **kwargs):
        """
        Compute numerical gradient using finite differences

        Args:
            state: Current simulation state
            var_value: Value at which to evaluate gradient
            iterations: Number of simulation steps
            eps: Small perturbation for finite difference

        Returns:
            tuple: (function value, gradient)
        """
        wrap = partial(numerical_diff.wrapper,
                      step_fun=self.step_function,
                      var_name=self.var_name,
                      agg_func=self.agg_function,
                      iter=iterations)

        center = wrap(var_value, state)
        rgt = wrap(var_value - eps, state)
        numerical_grad = (center - rgt)/(eps)

        return center, numerical_grad

class forward_diff(autodiff):
    """Forward-mode automatic differentiation implementation"""

    def g(self, state, var_value, iterations=1, eps=1e-9, **kwargs):
        """
        Compute gradient using forward-mode autodiff

        Args:
            state: Current simulation state
            var_value: Value at which to evaluate gradient
            iterations: Number of simulation steps

        Returns:
            gradient computed using JAX's forward-mode autodiff
        """
        wrap = partial(numerical_diff.wrapper,
                      step_fun=self.step_function,
                      var_name=self.var_name,
                      agg_func=self.agg_function,
                      iter=iterations)

        grad_forward = jacfwd(wrap, argnums=0)(var_value, state)
        return grad_forward

class jvp_diff(autodiff):
    """Jacobian-vector product implementation"""

    def g(self, state, var_value, iterations=1, **kwargs):
        """
        Compute gradient using JVP

        Args:
            state: Current simulation state
            var_value: Value at which to evaluate gradient
            iterations: Number of simulation steps

        Returns:
            tuple: (function value, gradient computed using JVP)
        """
        n_state = numerical_diff.set_var(self.var_name, state, var_value)
        tangent_state = n_state.get_tangeant(self.var_name)

        # Propagate tangents through iterations
        for i in range(iterations):
            next_state, tangent_state = jax.jvp(self.step_function, (n_state,), (tangent_state,))
        center, grad_jvp = jax.jvp(agg_sum, (n_state,), (tangent_state,))
        return center, grad_jvp

if __name__ =='__main__':
    # Initialize and warm up simulation
    acc = warmup_acc(300)
    acc.state._diagnostics = {}

    # Define aggregation function for gradient computation
    def agg_sum(state, key_sum='u', cv=slice(-5,-1,1)):
        """Compute mean squared difference of velocity field"""
        return (getattr(state.variables, key_sum)[:,:,:].mean() - 0)**2

    # Setup parameters for gradient computation
    step_function = acc.step
    agg_function = agg_sum
    var_name = 'r_bot'
    var = jnp.array(1e-5, dtype=jnp.float64)
    iteration_grad = 10

    # Compare different differentiation methods
    methods = [numerical_diff, forward_diff, jvp_diff]
    for ad_method in methods:
        ad = ad_method(step_function, agg_function, var_name)

        start_time = time.time()
        n_diff = ad.g(acc.state, var_value=var, iterations=1)
        print(f'{ad}: {n_diff}')
        n_diff_time = time.time() - start_time
        print(f'Time taken for {ad}: {n_diff_time:.4f} seconds')
