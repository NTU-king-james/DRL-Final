from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Example of running StarCraft2 with RLlib QMIX.

This assumes all agents are homogeneous. The agents are grouped and assigned
to the multi-agent QMIX policy. Note that the default hyperparameters for
RLlib QMIX are different from pymarl's QMIX.
"""

import argparse
from gym.spaces import Tuple

import ray
from ray import tune
from ray.tune.registry import register_env
from smac.examples.rllib.env import RLlibStarCraft2Env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--map-name", type=str, default="3m")
    args = parser.parse_args()

    def env_creator(smac_args):
        env = RLlibStarCraft2Env(**smac_args)
        agent_list = list(range(env._env.n_agents))
        grouping = {
            "group_1": agent_list,
        }
        obs_space = Tuple([env.observation_space for i in agent_list])
        act_space = Tuple([env.action_space for i in agent_list])
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        )

    ray.init()
    register_env("sc2_grouped", env_creator)

    tune.run(
        "QMIX",
        name="qmix_sc2",
        stop={"training_iteration": args.num_iters},
        config={
            "num_workers": args.num_workers,
            "env": "sc2_grouped",
            "env_config": {
                "map_name": args.map_name,
            },
        },
        local_dir="./results",
        checkpoint_freq=10,
        checkpoint_at_end=True,
    )
