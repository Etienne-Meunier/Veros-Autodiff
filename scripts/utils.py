from acc.acc import ACCSetup
from functools import partial
import abc
from tqdm import tqdm

def warmup_acc(steps_bf_check = 300, override_settings={}) :
    acc = ACCSetup(override=override_settings)
    acc.setup()
    with acc.state.variables.unlock()  :
        acc.state.variables.r_bot += 1e-5
    with acc.state.settings.unlock() :
        acc.state.settings.enable_tke = True
        acc.state.settings.enable_neutral_diffusion = True
    for step in tqdm(range(steps_bf_check)) :
        acc.step(acc.state)
    return acc

class autodiff() :
    def __init__(self, step_function, agg_function,  var_name) :
        """
            Computes derivative dL/dvar with L in R and var in R
            step_function is the function done n iterations
            agg_function is computed at the end to go from R^space -> R
            var_name : name of the variable in the state to differentiatiat w.r.t
        """
        self.agg_function = agg_function
        self.step_function = partial(autodiff.pure, step=step_function)
        self.var_name = var_name

    @staticmethod
    def pure(state, step) :
        """
            Convert the state function into a "pure step" copying the input state
        """
        n_state = state.copy()
        step(n_state)  # This is a function that modifies state object inplace
        return n_state

    @staticmethod
    def set_var(var_name, state, var_value):
        n_state = state.copy()
        vs = n_state.variables
        with n_state.variables.unlock():
            setattr(vs, var_name, var_value)
        return n_state


    @staticmethod
    def wrapper(var_value, state, step_fun, var_name, agg_func, iter):
        n_state = autodiff.set_var(var_name, state, var_value)

        for i in range(iter) :
            n_state = step_fun(n_state)

        return agg_func(n_state)


    @abc.abstractmethod
    def g(self, state, var_value, iterations=1, **kwargs) :
        """
            var_value : evaluation value for variable
            iterations : number of time to execute step_function
        Returns :
            output : agg_function([step_function(state)]*it) in R
            grad : (d ouput / d var_name |_var_value) in R
        """
        pass
