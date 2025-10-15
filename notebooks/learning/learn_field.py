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

<<<<<<< HEAD
=======
breakpoint()
for step in tqdm(range(10)) :
    acc.step(acc.state)
>>>>>>> 33192fb7ae1e12682fec40f362689860fa8b0d84

for step in tqdm(range(warmup_steps)) :
    acc.step(acc.state)

breakpoint()

# Build target

initial_state = acc.state.copy()
target_state = initial_state.copy()

for step in tqdm(range(n_iteration)) :
    acc.step(target_state)

target_field = getattr(target_state.variables, variable_to_perturb).copy()

# Perturb 
perturb_state = initial_state.copy()

<<<<<<< HEAD
# with perturb_state.variables.unlock() :
#     v = getattr(perturb_state.variables, variable_to_perturb)
#     setattr(perturb_state.variables, variable_to_perturb, v*1.01)
=======
with perturb_state.variables.unlock() :
    v = getattr(perturb_state.variables, variable_to_perturb)
    setattr(perturb_state.variables, variable_to_perturb, v*1.3)

field = getattr(perturb_state.variables, variable_to_perturb).copy()

>>>>>>> 33192fb7ae1e12682fec40f362689860fa8b0d84

field = getattr(perturb_state.variables, variable_to_perturb).copy() * 1.3


# Minimize
def agg_sum(state, target_state, key_sum = 'temp') :
    return ((getattr(state.variables, key_sum)[...,1] - getattr(target_state.variables, key_sum)[...,1])**2).mean()

step_function = acc.step
agg_function = lambda state : agg_sum(state, target_state, key_sum=variable_to_compare)

vjpm = vjp_grad_new(step_function, agg_function, variable_to_perturb)

loss_and_grad = jax.jit(lambda s, v: vjpm.g(s, v, iterations=n_iteration))



stats = []
<<<<<<< HEAD
pbar = tqdm(range(200))
    
=======
pbar = tqdm(range(2000))
>>>>>>> 33192fb7ae1e12682fec40f362689860fa8b0d84
for i in pbar:
    output_forward, gradients = loss_and_grad(perturb_state, field)
    distance = ((field - target_field) ** 2).mean()
    stats.append({'loss': output_forward, 'distance': distance})
<<<<<<< HEAD
    field -= 10.0 * gradients
=======
    field -= 0.1 * gradients
    
>>>>>>> 33192fb7ae1e12682fec40f362689860fa8b0d84
    pbar.set_postfix(loss=float(output_forward), distance=float(distance))
    if i%1000 == 0 : 
        print({'loss': output_forward, 'distance': distance})

<<<<<<< HEAD
jnp.
=======
    
print(stats)
>>>>>>> 33192fb7ae1e12682fec40f362689860fa8b0d84
