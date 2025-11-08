from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

from network import *
from config import *


def show_reward(log_path):
  with open(f'{log_path}/reward.txt', 'r') as file:
    rewards = file.read().split('\n')[:-1]

    reward_mean = [0]
    reward_low = [0]
    reward_high = [0]
    for reward in rewards:
      mean, std = reward.split(',')
      mean = float(mean)
      std = float(std)
      reward_mean.append(mean)
      reward_low.append(mean - 3*std)
      reward_high.append(mean + 3*std)

    # plt.close()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid(True)
    ax.set_xlabel('iteration')
    ax.set_ylabel('reward')
    num = np.arange(len(reward_mean))
    ax.plot(num, reward_mean, lw=2)
    ax.fill_between(num, reward_low, reward_high, alpha=0.5, color='aquamarine')
    plt.show()


def show_cartpole(log_path):
  seed = datetime.now().microsecond
  key = jax.random.key(seed)
  key, subkey = jax.random.split(key)
  actor = Actor(rngs=nnx.Rngs(subkey))
  checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
  load_model_params(checkpointer, f'{log_path}/model/actor', actor)
  actor.eval()

  cw = 0.13
  ch = 0.08
  xlim = (-1.1, 1.1)
  ylim = (-(l1+l2+0.1), l1+l2+0.1)

  # plt.close()
  fig1 = plt.figure("sim")
  ax1 = fig1.add_subplot()
  ax1.set_xlim(xlim)
  ax1.set_ylim(ylim)
  ax1.set_aspect('equal')
  ax1.grid(True)

  cart = Rectangle((0, 0), cw, ch, fc='Black')
  ax1.add_patch(cart)
  pole1, = ax1.plot([], [], lw=6, solid_capstyle='round', color='Cyan')
  pole2, = ax1.plot([], [], lw=6, solid_capstyle='round', color='Orange')

  plt.ion()
  plt.show()

  state = jnp.array([0.0, jnp.pi, jnp.pi, 0, 0, 0], jnp.float32)
  jitted_step = jax.jit(step)
  for t in range(151):
    if t % (int(action_step_cnt/render_step_cnt)) == 0:
      observation = get_observation(state)
      mu, sig, w = actor(observation)
      k = jnp.argmax(w)
      action = action_max * jnp.tanh(mu[k])
    next_state = jitted_step(state, action, render_step_cnt)
    print(action, get_reward(state, action, next_state, False))
    state = next_state
    x = state[0]
    th1 = state[1]
    th2 = state[2]
    p0 = (x, ch/2)
    p1 = (x-l1*jnp.sin(th1), ch/2+l1*jnp.cos(th1))
    p2 = (x-l1*jnp.sin(th1)-l2*jnp.sin(th2), ch/2+l1*jnp.cos(th1)+l2*jnp.cos(th2))
    cart.set_xy((x - cw/2, 0))
    pole1.set_data((p0[0], p1[0]), (p0[1], p1[1]))
    pole2.set_data((p1[0], p2[0]), (p1[1], p2[1]))
    plt.draw()
    plt.pause(render_interval)
    if jnp.abs(x) > 1 :
      state = jnp.array([0, jnp.pi, jnp.pi, 0, 0, 0], jnp.float32)


show_cartpole('./log')
show_reward('./log')