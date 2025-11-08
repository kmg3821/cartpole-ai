from config import *
from environment import *


class LinearBlock(nnx.Module):
  def __init__(self, in_dim, out_dim, *, rngs):
    self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)
  
  def __call__(self, x):
    h = self.linear(x)
    h = nnx.relu(h)
    return h


class Actor(nnx.Module):
  def __init__(self, rngs):
    self.linear_common = nnx.Sequential(
      LinearBlock(observation_dim,256, rngs=rngs),
      LinearBlock(256,256, rngs=rngs),
    )
    self.linear_mu = nnx.Sequential(
      LinearBlock(256,256, rngs=rngs),
      nnx.Linear(256,modal_num*action_dim, rngs=rngs)
    )
    self.linear_sig = nnx.Sequential(
      LinearBlock(256,256, rngs=rngs),
      nnx.Linear(256,modal_num*action_dim, rngs=rngs)
    )
    self.linear_w = nnx.Sequential(
      LinearBlock(256,256, rngs=rngs),
      nnx.Linear(256,modal_num, rngs=rngs)
    )

  def __call__(self, observation):
    h = self.linear_common(observation)   

    mu = self.linear_mu(h)
    if observation.ndim == 1: mu = jnp.reshape(mu, (modal_num, action_dim))
    else: mu = jnp.reshape(mu, (-1, modal_num, action_dim))

    sig = self.linear_sig(h)
    sig = jnp.exp(jnp.clip(sig, min=-20, max=2))
    if observation.ndim == 1: sig = jnp.reshape(sig, (modal_num, action_dim))
    else: sig = jnp.reshape(sig, (-1, modal_num, action_dim))

    w = self.linear_w(h)
    w = jax.nn.softmax(w, axis=-1)
    if observation.ndim == 1: w = jnp.reshape(w, (modal_num,))
    else: w = jnp.reshape(w, (-1, modal_num))

    return mu, sig, w


class Critic(nnx.Module):
  def __init__(self, rngs):
    self.linear = nnx.Sequential(
      LinearBlock(observation_dim+action_dim,256, rngs=rngs),
      LinearBlock(256,256, rngs=rngs),
      LinearBlock(256,256, rngs=rngs),
      nnx.Linear(256,1, rngs=rngs)
    )

  def __call__(self, observation, action):
    h = jnp.hstack([observation, action])
    h = self.linear(h)
    if observation.ndim == 1: h = jnp.reshape(h, ())
    else: h = jnp.reshape(h, (-1,))
    return h


class Alpha(nnx.Module):
  def __init__(self):
    self.logalpha = nnx.Param(jnp.log(0.2))

  def __call__(self):
    return jnp.exp(self.logalpha)


def vote_critic(critic1, critic2, observation, action):
  q1 = critic1(observation, action)
  q2 = critic2(observation, action)
  if observation.ndim == 1: q = jnp.min(jnp.array([q1, q2]))
  else: q = jnp.min(jnp.vstack([q1, q2]), axis=0)
  return q


@nnx.jit
def train_critic(critic1_optim, critic2_optim, actor, critic1_target, critic2_target, 
                 alpha, observation, action, reward, next_observation, done, key):
  graphdef, _, rngs = nnx.split(critic1_optim.model, nnx.Param, nnx.RngState)
  critic1_target_model = nnx.merge(graphdef, critic1_target, rngs)
  critic1_target_model.eval()

  graphdef, _, rngs = nnx.split(critic2_optim.model, nnx.Param, nnx.RngState)
  critic2_target_model = nnx.merge(graphdef, critic2_target, rngs)
  critic2_target_model.eval()

  actor.eval()
  alpha.eval()

  next_action, logprob = get_action(actor, next_observation, key, has_logprob=True)
  q = vote_critic(critic1_target_model, critic2_target_model, next_observation, next_action)
  target = reward + (1 - done)*gamma*(q - alpha()*logprob)
  assert target.shape == q.shape

  def loss(critic):
    q = critic(observation, action)
    assert target.shape == q.shape
    params = nnx.state(critic, nnx.Param)
    l2_reg = jax.tree.reduce(lambda sum, x: sum + jnp.sum(x**2), params, initializer=0)
    Jq = jnp.sum(0.5*(q - target)**2).mean() + 1e-4*l2_reg
    return Jq

  critic1_optim.model.train()
  loss1, grad = nnx.value_and_grad(loss)(critic1_optim.model)
  critic1_optim.update(grad)

  critic2_optim.model.train()
  loss2, grad = nnx.value_and_grad(loss)(critic2_optim.model)
  critic2_optim.update(grad)

  return loss1, loss2


@nnx.jit
def train_actor_alpha(actor_optim, alpha_optim, critic1, critic2, observation, key):
  critic1.eval()
  critic2.eval()
  alpha_optim.model.eval()
  
  def actor_loss(actor, alpha, key):
    action, logprob, mu = get_action(actor, observation, key, has_logprob=True, has_mu=True)
    q = vote_critic(critic1, critic2, observation, action)
    assert logprob.shape == q.shape
    params = nnx.state(actor, nnx.Param)
    l2_reg = jax.tree.reduce(lambda sum, x: sum + jnp.sum(x**2), params, initializer=0)
    Jpi = jnp.sum(alpha()*logprob - q).mean() + 1e-4*l2_reg
    # Jpi += 1e-3*jnp.sum(jnp.exp(-(jnp.expand_dims(mu,1) - jnp.expand_dims(mu,2))**2))
    return Jpi, logprob
  
  actor_optim.model.train()
  (actor_loss, logprob), grad = nnx.value_and_grad(actor_loss, has_aux=True)(actor_optim.model, alpha_optim.model, key)
  actor_optim.update(grad)

  def alpha_loss(alpha):
    Jalpha = (alpha() * (-logprob + action_dim)).mean()
    return Jalpha

  alpha_optim.model.train()
  alpha_loss, grad = nnx.value_and_grad(alpha_loss)(alpha_optim.model)
  alpha_optim.update(grad)

  return actor_loss, alpha_loss


@nnx.jit
def soft_update(critic_target, critic):
  critic_target = jax.tree.map(lambda x, y: rho*x + (1-rho)*y, critic_target, nnx.state(critic, nnx.Param))
  return critic_target


def save_model_params(checkpointer, savepath, model, usekey=False):
  savepath = os.path.abspath(savepath)
  if usekey == False:
    state = nnx.state(model, nnx.Param)
  else:
    key, state = nnx.state(model, nnx.RngKey, ...)
    key = jax.tree.map(jax.random.key_data, key)
    checkpointer.save(savepath + '/key', key, force=True)
  checkpointer.save(savepath + '/state', state, force=True)

    
def load_model_params(checkpointer, loadpath, model, usekey=False):
  def set_sharding(x):
      return x.update(sharding=NSpec(Mesh(device, ('dev',)), PSpec()))
  
  loadpath = os.path.abspath(loadpath)

  if usekey == False:
    state = nnx.state(model, nnx.Param)  
  else:
    key, state = nnx.state(model, nnx.RngKey, ...)
    key = jax.tree.map(jax.random.key_data, key)
    struct_key = nnx.eval_shape(lambda: key)
    struct_key = jax.tree.map(set_sharding, struct_key)
    key = checkpointer.restore(loadpath + '/key', args=ocp.args.StandardRestore(struct_key))
    key = jax.tree.map(jax.random.wrap_key_data, key)

  struct_state = nnx.eval_shape(lambda: state)
  struct_state = jax.tree.map(set_sharding, struct_state)
  state = checkpointer.restore(loadpath + '/state', args=ocp.args.StandardRestore(struct_state))

  if usekey == False:
    nnx.update(model, state)
  else:
    nnx.update(model, key, state)






