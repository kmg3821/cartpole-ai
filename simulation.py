from config import *
from network import *
from environment import *


@nnx.jit
@partial(nnx.shard_map, mesh=Mesh(device, ('batch',)),
         in_specs=(PSpec(), PSpec('batch'), PSpec('batch')), 
         out_specs=PSpec('batch'), check_rep=False)
@partial(nnx.vmap, in_axes=(None,0,0), out_axes=0)
def simulate(actor, init_state, key):
  actor.eval()

  def cond(val):
    cnt, episode, terminated, key = val
    return (cnt < episode_cnt) & (terminated == False)

  def body(val):
    cnt, episode, terminated, key = val

    key, subkey = jax.random.split(key)
    observation = get_observation(episode["state"], cnt)
    action = get_action(actor, observation, subkey)

    state = episode["state"][cnt]
    next_state = step(state, action, action_step_cnt)
    terminated = is_terminated(next_state)
    reward = get_reward(state, action, next_state, terminated)

    episode["action"] = episode["action"].at[cnt].set(action)
    episode["reward"] = episode["reward"].at[cnt].set(reward)
    episode["state"] = episode["state"].at[cnt+1].set(next_state)

    return cnt+1, episode, terminated, key
  
  episode = {
    "state" : jnp.zeros((episode_cnt+1, state_dim)),
    "action" : jnp.zeros((episode_cnt, action_dim)),
    "reward" : jnp.zeros((episode_cnt,)),
  }
  episode["state"] = episode["state"].at[0].set(init_state)
  cnt, episode, terminated, key = nnx.while_loop(cond, body, (0, episode, False, key))

  return cnt, episode, terminated


@nnx.jit
@partial(nnx.shard_map, mesh=Mesh(device, ('batch',)),
         in_specs=(PSpec('batch'), PSpec('batch')), 
         out_specs=PSpec('batch'), check_rep=False)
@partial(nnx.vmap, in_axes=(0,0), out_axes=0)
def simulate_randomly(init_state, key):
  def cond(val):
    cnt, episode, terminated, key = val
    return (cnt < episode_cnt) & (terminated == False)

  def body(val):
    cnt, episode, terminated, key = val

    state = episode["state"][cnt]
    key, subkey = jax.random.split(key)
    action = get_random_action(subkey)

    next_state = step(state, action, action_step_cnt)
    terminated = is_terminated(next_state)
    reward = get_reward(state, action, next_state, terminated)

    episode["action"] = episode["action"].at[cnt].set(action)
    episode["reward"] = episode["reward"].at[cnt].set(reward)
    episode["state"] = episode["state"].at[cnt+1].set(next_state)

    return cnt+1, episode, terminated, key
  
  episode = {
    "state" : jnp.zeros((episode_cnt+1, state_dim)),
    "action" : jnp.zeros((episode_cnt, action_dim)),
    "reward" : jnp.zeros((episode_cnt,)),
  }
  episode["state"] = episode["state"].at[0].set(init_state)
  cnt, episode, terminated, key = nnx.while_loop(cond, body, (0, episode, False, key))

  return cnt, episode, terminated


def init_replaybuffer(replaybuffer, key):
  key, subkey = jax.random.split(key)
  init_state = get_init_state(init_batch_size, subkey)

  for ith in range(math.ceil(replaybuffer_size/init_batch_size)) :
    key, *subkey = jax.random.split(key, init_batch_size+1)
    cnt, episode, terminated = simulate_randomly(init_state, jnp.asarray(subkey))

    tail = replaybuffer["tail"]
    idx = (tail + np.arange(init_batch_size)) % replaybuffer_size

    replaybuffer["tail"] = (tail + init_batch_size) % replaybuffer_size
    replaybuffer["state"][idx] = episode["state"]
    replaybuffer["action"][idx] = episode["action"]
    replaybuffer["reward"][idx] = episode["reward"]
    replaybuffer["cnt"][idx] = cnt
    replaybuffer["terminated"][idx] = terminated

    print(f"Fill: {init_batch_size * (ith+1)}/{replaybuffer_size}", end="\r")
  print("\n" + "="*30)


def load_replaybuffer(loadpath=None):
  if loadpath == None :
    replaybuffer = {
      "state": np.zeros((replaybuffer_size, episode_cnt+1, state_dim), np.float32), 
      "action": np.zeros((replaybuffer_size, episode_cnt, action_dim), np.float32), 
      "reward": np.zeros((replaybuffer_size, episode_cnt,), np.float32), 
      "terminated" : np.zeros((replaybuffer_size,), np.int8),
      "cnt" : np.zeros((replaybuffer_size,), np.int32),
      "tail": np.int32(0)
    }
  else:
    data = np.load(loadpath)
    replaybuffer = {
      "state": np.array(data["state"], np.float32),
      "action": np.array(data["action"], np.float32),
      "reward": np.array(data["reward"], np.float32),
      "terminated": np.array(data["terminated"], np.int8),
      "cnt": np.array(data["cnt"], np.int32),
      "tail": np.int32(data["tail"])
    }
  return replaybuffer


def save_replaybuffer(replaybuffer, savepath):
  np.savez_compressed(savepath, 
    state=replaybuffer["state"],
    action=replaybuffer["action"],
    reward=replaybuffer["reward"],
    terminated=replaybuffer["terminated"],
    cnt=replaybuffer["cnt"],
    tail=replaybuffer["tail"])


def add_episodes_to_replaybuffer(replaybuffer, actor, key):
  key, subkey = jax.random.split(key)
  init_state = get_init_state(batch_size, subkey)

  key = jax.random.split(key, batch_size)
  cnt, episode, terminated = simulate(actor, init_state, jnp.asarray(key))
  
  tail = replaybuffer["tail"]
  idx = (tail + np.arange(batch_size)) % replaybuffer_size

  replaybuffer["tail"] = (tail + batch_size) % replaybuffer_size
  replaybuffer["state"][idx] = episode["state"]
  replaybuffer["action"][idx] = episode["action"]
  replaybuffer["reward"][idx] = episode["reward"]
  replaybuffer["cnt"][idx] = cnt
  replaybuffer["terminated"][idx] = terminated

  return episode["reward"]

