import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
# os.environ['XLA_FLAGS'] = '--xla_gpu_first_collective_call_terminate_timeout_seconds=7200'

from simulation import *
from config import *

jax.config.update("jax_compilation_cache_dir", f"jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

log_dir = "./log"
print_interval = 50
save_interval = 100
new_data_iteration = 5000
seed = 0

os.makedirs(log_dir, exist_ok=True)

if os.path.exists(f'{log_dir}/np_key_state.pkl') == True:
  with open(f'{log_dir}/np_key_state.pkl', 'rb') as file:
    key_state = pickle.load(file)
    np.random.set_state(key_state)
else :
  np.random.seed(seed)

if os.path.exists(f'{log_dir}/jax_key_data.npy') == True:
  key_data = jnp.load(f"{log_dir}/jax_key_data.npy")
  key = jax.random.wrap_key_data(key_data)
else :
  key = jax.random.key(seed+1)

checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

key, subkey = jax.random.split(key)
actor_optim = nnx.ModelAndOptimizer(Actor(nnx.Rngs(subkey)), optax.adam(lr))

key, subkey = jax.random.split(key)
critic1_optim = nnx.ModelAndOptimizer(Critic(nnx.Rngs(subkey)), optax.adam(lr))
critic1_target = nnx.state(critic1_optim.model, nnx.Param)

key, subkey = jax.random.split(key)
critic2_optim = nnx.ModelAndOptimizer(Critic(nnx.Rngs(subkey)), optax.adam(lr))
critic2_target = nnx.state(critic2_optim.model, nnx.Param)

alpha_optim = nnx.ModelAndOptimizer(Alpha(), optax.adam(lr))

if os.path.exists(f'{log_dir}/replaybuffer.npz') == True:
  replaybuffer = load_replaybuffer(f'{log_dir}/replaybuffer.npz')
else:
  replaybuffer = load_replaybuffer()
  key, subkey = jax.random.split(key)
  init_replaybuffer(replaybuffer, subkey)

if os.path.exists(f'{log_dir}/model') == True:
  load_model_params(checkpointer, f'{log_dir}/model/actor', actor_optim.model)
  load_model_params(checkpointer, f'{log_dir}/model/critic1', critic1_optim.model)
  load_model_params(checkpointer, f'{log_dir}/model/critic2', critic2_optim.model)
  load_model_params(checkpointer, f'{log_dir}/model/alpha', alpha_optim.model)

reward_history = list()
for ith in range(new_data_iteration):
  data0, done0 = sample_train_data(replaybuffer)

  for jth in range(update_iteration):
    if (jth % print_interval == 0) or (jth == update_iteration-1) : print(f"[{ith}][{jth}]")

    observation = data0["observation"][jth]
    action = data0["action"][jth]
    reward = data0["reward"][jth]
    next_observation = data0["next_observation"][jth]
    done = done0[jth]
    
    key, subkey = jax.random.split(key)
    loss1, loss2 = train_critic(critic1_optim, critic2_optim, actor_optim.model, critic1_target, critic2_target, 
                                alpha_optim.model, observation, action, reward, next_observation, done, subkey)
    key, subkey = jax.random.split(key)
    actor_loss, alpha_loss = train_actor_alpha(actor_optim, alpha_optim, 
                                               critic1_optim.model, critic2_optim.model, observation, subkey)
    
    critic1_target = soft_update(critic1_target, critic1_optim.model)
    critic2_target = soft_update(critic2_target, critic2_optim.model)

    if (jth % print_interval == 0) or (jth == update_iteration-1) : 
      print(f"critic loss: {loss1}, {loss2}")
      print(f"actor loss: {actor_loss}")
      print(f"alpha loss: {alpha_loss}")
      print("="*30)

  key, subkey = jax.random.split(key)
  reward = add_episodes_to_replaybuffer(replaybuffer, actor_optim.model, subkey)
  reward_sum = np.sum(reward, axis=1)
  reward_mean = np.mean(reward_sum)
  reward_std = np.std(reward_sum)
  reward_history.append(f"{reward_mean},{reward_std}")

  print(f"reward: {reward_mean} ({reward_std})")
  alpha_optim.model.eval()
  print(f"alpha: {alpha_optim.model()}")
  print("="*30)

  if (ith % save_interval == 0) or (ith == new_data_iteration-1):
    save_model_params(checkpointer, f'{log_dir}/model/actor', actor_optim.model)
    save_model_params(checkpointer, f'{log_dir}/model/critic1', critic1_optim.model)
    save_model_params(checkpointer, f'{log_dir}/model/critic2', critic2_optim.model)
    save_model_params(checkpointer, f'{log_dir}/model/alpha', alpha_optim.model)
    save_replaybuffer(replaybuffer, f'{log_dir}/replaybuffer.npz')
    jnp.save(f"{log_dir}/jax_key_data.npy", jax.random.key_data(key))
    with open(f'{log_dir}/np_key_state.pkl', 'wb') as file:
      pickle.dump(np.random.get_state(), file)
    with open(f'{log_dir}/reward.txt', 'a') as file:
      file.write("\n".join(reward_history) + '\n')
    reward_history.clear()

checkpointer.close()

