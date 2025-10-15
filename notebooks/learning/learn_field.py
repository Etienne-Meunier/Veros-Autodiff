from __init__ import PRP; import sys
sys.path.append(PRP)

from scripts.grad_compare import *

from setups.acc.acc_learning import ACCSetup
from tqdm import tqdm
import gc 

warmup_steps = 1000 
n_iteration = 3 # Number of steps iterations for training
variable_to_perturb = 'temp' # Perturb a variable in the initial state
variable_to_compare = 'temp' # Compare between perturbed and target


# Spin-up 
acc = ACCSetup()
acc.setup()


for step in tqdm(range(warmup_steps)) :
    acc.step(acc.state)


# Build target

initial_state = acc.state.copy()
target_state = initial_state.copy()

for step in tqdm(range(n_iteration)) :
    acc.step(target_state)

target_field = getattr(target_state.variables, variable_to_perturb).copy()

# Perturb 
perturb_state = initial_state.copy()

# with perturb_state.variables.unlock() :
#     v = getattr(perturb_state.variables, variable_to_perturb)
#     setattr(perturb_state.variables, variable_to_perturb, v*1.01)

field = getattr(perturb_state.variables, variable_to_perturb).copy() * 1.3


# Minimize
def agg_sum(state, target_state, key_sum = 'temp') :
    return ((getattr(state.variables, key_sum)[...,1] - getattr(target_state.variables, key_sum)[...,1])**2).mean()

step_function = acc.step
agg_function = lambda state : agg_sum(state, target_state, key_sum=variable_to_compare)

vjpm = vjp_grad_new(step_function, agg_function, variable_to_perturb)

loss_and_grad = jax.jit(lambda s, v: vjpm.g(s, v, iterations=n_iteration))



stats = []
pbar = tqdm(range(200))
    
for i in pbar:
    output_forward, gradients = loss_and_grad(perturb_state, field)
    distance = ((field - target_field) ** 2).mean()
    stats.append({'loss': output_forward, 'distance': distance})
    field -= 10.0 * gradients
    pbar.set_postfix(loss=float(output_forward), distance=float(distance))
    if i%1000 == 0 : 
        print({'loss': output_forward, 'distance': distance})

jnp.