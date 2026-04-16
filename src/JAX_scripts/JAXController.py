import jax
import jax.numpy as jnp
 
class JAXPolicyController:
    def __init__(self, jax_model, action_range, deterministic=True):
        """
        A simple JAX controller wrapping the policy network.
        :param jax_model: Callable JAX policy function
        :param action_range: array-like, max action in each dimension
        :param deterministic: if True, uses mean action; else samples stochastically
        """
        self.model = jax_model
        self.action_range = jnp.array(action_range)
        self.deterministic = deterministic
        self.key = jax.random.PRNGKey(0)  # PRNG for stochastic actions
 
    def get_action(self, state):
        """
        Get an action from the policy given a state.
        :param state: array-like, e.g., [x, y, time]
        :return: action array
        """
        state = jnp.array(state).reshape(1, -1)  # add batch dimension
 
        mean, log_std = self.model(state)
 
        if self.deterministic:
            action = jnp.tanh(mean) * self.action_range
        else:
            self.key, subkey = jax.random.split(self.key)
            std = jnp.exp(log_std)
            z = jax.random.normal(subkey, shape=mean.shape)
            action_0 = jnp.tanh(mean + std * z)
            action = self.action_range * action_0
 
        return jnp.squeeze(action)  # remove batch dim