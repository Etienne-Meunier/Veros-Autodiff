from __init__ import PRP; import sys
sys.path.append(PRP)

from scripts.grad_compare import *

from setups.acc.acc_learning import ACCSetup
from tqdm import tqdm


n_iteration = 3 # Number of steps iterations for training
variable_to_perturb = 'temp' # Perturb a variable in the initial state
variable_to_compare = 'temp' # Compare between perturbed and target


# Spin-up 
acc = ACCSetup()
acc.setup()

for step in tqdm(range(300)) :
    acc.step(acc.state)



# Build target

initial_state = acc.state.copy()
target_state = initial_state.copy()

for step in tqdm(range(n_iteration)) :
    acc.step(target_state)

target_field = getattr(target_state.variables, variable_to_perturb).copy()


# Perturb 
perturb_state = initial_state.copy()

with perturb_state.variables.unlock() :
    v = getattr(perturb_state.variables, variable_to_perturb)
    getattr(perturb_state.variables, variable_to_perturb, v*1.3)

field = getattr(perturb_state.variables, variable_to_perturb).copy()




# Minimize
def agg_sum(state, target_state, key_sum = 'temp') :
    return ((getattr(state.variables, key_sum)[...,1] - getattr(target_state.variables, key_sum)[...,1])**2).mean()

step_function = acc.step
agg_function = lambda state : agg_sum(state, target_state, key_sum=variable_to_compare)

vjpm = vjp_grad(step_function, agg_function, variable_to_perturb)

stats = []
pbar = tqdm(range(200))
for i in pbar:
    output_forward, gradients = vjpm.g(perturb_state, var_value=field, iterations=n_iteration, var_name=variable_to_perturb)
    distance = ((field - target_field) ** 2).mean()
    stats.append({'loss': output_forward, 'distance': distance})
    field -= 1.0 * gradients
    
    pbar.set_postfix(loss=float(output_forward), distance=float(distance))

    