import jax
from jax import numpy as jnp


try : device = jax.devices('tpu')
except: 
  try : device = jax.devices('gpu') 
  except : device = jax.devices('cpu')
# device = jax.devices('cpu')


# simulation parameters
m0 = 1.0
m1 = 0.15
m2 = 0.15
l1 = 0.2
l2 = 0.2
g = 9.81
action_max = jnp.array([5])

modal_num = 3
state_dim = 6
observation_dim = 6
action_dim = 1

render_interval = 0.05
action_interval = 0.1
step_interval = 0.01
end_time = 12.0

render_step_cnt = int(render_interval/step_interval)
action_step_cnt = int(action_interval/step_interval)
end_step_cnt = int(end_time/step_interval)
episode_cnt = int(end_time/action_interval)
assert (render_step_cnt >= 1) and (abs((render_interval/step_interval) - render_step_cnt) < 1e-12)
assert (action_step_cnt >= 1) and (abs((action_interval/step_interval) - action_step_cnt) < 1e-12)
assert (end_step_cnt >= 1) and (abs((end_time/step_interval) - end_step_cnt) < 1e-12)
assert (episode_cnt >= 1) and (abs((end_time/action_interval) - episode_cnt) < 1e-12)


# training/replaybuffer parameters
init_batch_size = 131072
batch_size = 4096
replaybuffer_size = 100000
minibatch_size = 1024
lr = 3e-4
gamma = 0.99
rho = 0.995
update_iteration = 500