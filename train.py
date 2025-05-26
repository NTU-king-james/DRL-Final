# train.py
import ray
from ray import tune
from ray.rllib.algorithms import QMixConfig
from smac_env import RLLibSMACEnv
from ray.tune.registry import register_env

def env_creator(env_config):
    return RLLibSMACEnv(env_config)

if __name__ == "__main__":
    ray.init()

    # 註冊自定義 SMAC 環境
    register_env("smac_custom", env_creator)

    # 建立 QMix 配置
    config = (
        QMixConfig()
        .environment(env="smac_custom", env_config={"map_name": "8m"})
        .training(
            mixer="qmix",
            train_batch_size=32,
        )
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=1)
    )

    tune.run(
        "QMix",
        name="QMix-SMAC-8m",
        stop={"training_iteration": 100},
        config=config.to_dict(),
        checkpoint_at_end=True,
    )