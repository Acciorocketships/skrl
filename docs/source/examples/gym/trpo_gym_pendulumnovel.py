import gym

import torch
import torch.nn as nn
import numpy as np

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.torch import wrap_env


# Define the models (stochastic and deterministic models) for the agent using mixins.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# Gym environment observation wrapper used to mask velocity. Adapted from rl_zoo3 (rl_zoo3/wrappers.py)
class NoVelocityWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        # observation: x, y, angular velocity
        return observation * np.array([1, 1, 0])

gym.envs.registration.register(id="PendulumNoVel-v1", entry_point=lambda: NoVelocityWrapper(gym.make("Pendulum-v1")))

# Load and wrap the Gym environment
env = gym.vector.make("PendulumNoVel-v1", num_envs=4, asynchronous=False)
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# TRPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.trpo.html#spaces-and-models
models_trpo = {}
models_trpo["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models_trpo["value"] = Value(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.trpo.html#configuration-and-hyperparameters
cfg_trpo = TRPO_DEFAULT_CONFIG.copy()
cfg_trpo["rollouts"] = 1024  # memory_size
cfg_trpo["learning_epochs"] = 10
cfg_trpo["mini_batches"] = 32
cfg_trpo["discount_factor"] = 0.99
cfg_trpo["lambda"] = 0.95
cfg_trpo["learning_rate"] = 1e-3
cfg_trpo["grad_norm_clip"] = 0.5
cfg_trpo["state_preprocessor"] = RunningStandardScaler
cfg_trpo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_trpo["value_preprocessor"] = RunningStandardScaler
cfg_trpo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 500 and 5000 timesteps respectively
cfg_trpo["experiment"]["write_interval"] = 500
cfg_trpo["experiment"]["checkpoint_interval"] = 5000

agent_trpo = TRPO(models=models_trpo,
                  memory=memory,
                  cfg=cfg_trpo,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_trpo)

# start training
trainer.train()
