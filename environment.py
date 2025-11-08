from config import *
from network import *


def get_random_action(key):
  action =  action_max * jax.random.uniform(key, (action_dim,), minval=-1, maxval=1)
  return action


def get_action(actor, observation, key, *, has_logprob=False, has_mu=False, deterministic=False):
  def process(actor, observation, key):
    mu, sig, w = actor(observation)
    if deterministic == True:
      k = jnp.argmax(w)
      sample = mu[k]
    else:
      key, subkey = jax.random.split(key)
      k = jax.random.choice(subkey, modal_num, p=w)
      key, subkey = jax.random.split(key)
      sample = jax.random.normal(subkey, (action_dim,)) * sig[k] + mu[k]
    action = action_max * jnp.tanh(sample)

    if has_logprob == True :
      prob = jnp.prod(jsp.stats.norm.pdf(sample, mu, sig), axis=-1)
      assert w.shape == prob.shape
      logprob = jnp.log(jnp.sum(w * prob))
      logprob -= jnp.sum(jnp.log(4*action_max) - 2*(sample + jax.nn.softplus(-2*sample)))
      if has_mu == True: return action, logprob, mu
      else: return action, logprob
    else:
      if has_mu == True: return action, mu
      else: return action
  
  if observation.ndim == 1: 
    return process(actor, observation, key)
  else:
    key = jax.random.split(key, observation.shape[0])
    return nnx.vmap(process, in_axes=(None,0,0))(actor, observation, key)


def normalize_angle(th):
  th = (th + jnp.pi) % (2*jnp.pi) - jnp.pi
  return th


@partial(jax.jit, static_argnames=['num'])
def get_init_state(num, key):
  num1 = num // 2
  num2 = num - num1
  key, subkey = jax.random.split(key)
  th1 = jax.random.uniform(subkey, (num1, 2), minval=-1, maxval=1) * jnp.pi
  key, subkey = jax.random.split(key)
  th2 = jax.random.uniform(subkey, (num2, 2), minval=-1, maxval=1) * jnp.pi/6
  th = jnp.vstack([th1, th2])
  init_state = jnp.zeros((num, state_dim)).at[:,1:3].set(th)

  key, subkey = jax.random.split(key)
  dx1 = jax.random.uniform(subkey, (num1,), minval=-1, maxval=1) * 0.9
  key, subkey = jax.random.split(key)
  dx2 = jax.random.uniform(subkey, (num2,), minval=-1, maxval=1) * 0.3
  dx = jnp.hstack([dx1, dx2])
  init_state = init_state.at[:,0].set(dx)

  return init_state


def step(state0, action, num) :
  def dynamics(state):
    F = action[0]
    th1 = state[1]
    th2 = state[2]
    xdot = state[3]
    th1dot = state[4]
    th2dot = state[5]

    A = jnp.array([
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, m0+m1+m2, -(m1/2+m2)*l1*jnp.cos(th1), -m2*l2/2*jnp.cos(th2)],
      [0, 0, 0, (m1/2+m2)*l1*jnp.cos(th1), -(m1/3+m2)*l1**2, -m2*l1*l2/2*jnp.cos(th1-th2)],
      [0, 0, 0, m2*l2/2*jnp.cos(th2), -m2*l1*l2/2*jnp.cos(th1-th2), -m2*l2**2/3]])

    B = jnp.array([xdot, th1dot, th2dot,
      F - (m1/2+m2)*l1*jnp.sin(th1)*th1dot**2 - m2*l2/2*jnp.sin(th2)*th2dot**2,
      m1*l1*l2/2*jnp.sin(th1-th2)*th2dot**2 - (m1/2+m2)*g*l1*jnp.sin(th1),
      -m2*l1*l2/2*jnp.sin(th1-th2)*th1dot**2 - m2*g*l2/2*jnp.sin(th2)])
    
    return jnp.linalg.solve(A, B)
  
  def body(i, state): # runge-kutta 4th
    k1 = dynamics(state)
    k2 = dynamics(state + k1*step_interval/2)
    k3 = dynamics(state + k2*step_interval/2)
    k4 = dynamics(state + k3*step_interval)
    return state + (k1 + 2*k2 + 2*k3 + k4)*step_interval/6
  
  state = jax.lax.fori_loop(0, num, body, state0)
  state = state.at[1:3].set(normalize_angle(state[1:3]))
  return state


def get_reward(state, control, next_state, terminated):
  Fmax = action_max[0]
  F = control[0]
  x = state[0]
  th1 = state[1]
  th2 = state[2]
  xdot = state[3]
  th1dot = state[4]
  th2dot = state[5]

  reward = 10 -2*(1 - jnp.cos(th1)) -2*(1 - jnp.cos(th2)) -4*jnp.abs(x)**2\
            -1e-3*(th1dot/jnp.pi)**2 -1e-2*(th2dot/jnp.pi)**2 -1e-1*jnp.abs(F/Fmax)**2 \
            +2*((jnp.abs(th1) < jnp.pi/180) & (jnp.abs(th2) < jnp.pi/180))\
            +3*((jnp.abs(th1) < jnp.pi/180) & (jnp.abs(th2) < jnp.pi/180) & (jnp.abs(x) < 0.01))\
            +5*((jnp.abs(th1) < jnp.pi/180) & (jnp.abs(th2) < jnp.pi/180) & (jnp.abs(x) < 0.01) & (jnp.abs(F) < 0.01))
  reward = jnp.where(terminated, -10, reward)
  return reward*0.1


def is_terminated(state):
  x = state[0]
  terminated = (jnp.abs(x) > 1)
  return terminated


def get_observation(state, cnt=None):
  if cnt == None:
    observation = state
  else:
    observation = state[cnt]

  return observation


def sample_train_data(replaybuffer):
  cnt = replaybuffer["cnt"]
  accum = np.cumsum(cnt, dtype=np.int32)
  pos = np.random.choice(accum[-1], (update_iteration*minibatch_size,), replace=True)
  buf_idx = np.searchsorted(accum, pos, side='right')
  cnt_idx = cnt[buf_idx] - (accum[buf_idx] - pos)

  observation = replaybuffer["state"][buf_idx, cnt_idx].reshape((update_iteration, minibatch_size, state_dim))
  next_observation = replaybuffer["state"][buf_idx, cnt_idx+1].reshape((update_iteration, minibatch_size, state_dim))

  train_data = {
    "observation" : observation,
    "next_observation" : next_observation,
    "action" : replaybuffer["action"][buf_idx, cnt_idx].reshape((update_iteration, minibatch_size, action_dim)),
    "reward" : replaybuffer["reward"][buf_idx, cnt_idx].reshape((update_iteration, minibatch_size)),
  }
  done = jnp.array((replaybuffer["terminated"][buf_idx] == 1) 
                   & (cnt[buf_idx] == cnt_idx+1)).reshape((update_iteration, minibatch_size))
  
  return train_data, done
