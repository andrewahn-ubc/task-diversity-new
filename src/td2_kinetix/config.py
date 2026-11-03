"""Immutable protocol for the paper-shaped Kinetix experiment.

The paper specifies the task construction, pool sizes, distribution length,
and number of seeds, but not a separate Kinetix model/PPO table.  The learner
settings below are consequently the defaults from the official Kinetix
``v0.1.0`` reproduction tag.  Keeping them in one typed object makes the
scientific protocol auditable and gives every checkpoint/data bundle one
stable identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


KINETIX_COMMIT = "eed109cb40d777f5ca62eddda134d74d92543870"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KinetixProtocol:
    # Requested continual-learning protocol.
    learner_seeds: tuple[int, ...] = (0, 1, 2)
    diversity_levels: tuple[int, ...] = (1, 16, 64, 256, 65_536)
    num_distributions: int = 7
    timesteps_per_distribution: int = 100_000_000
    heldout_tasks: int = 100
    heldout_timesteps: int = 100_000_000

    # Paper task construction (small Kinetix preset).
    num_polygons: int = 5
    num_circles: int = 2
    num_joints: int = 1
    num_thrusters: int = 1
    frame_skip: int = 2
    max_episode_steps: int = 256
    dense_reward_scale: float = 2.0
    no_op_filter_steps: int = 64
    data_root_seed: int = 2_026_110_3

    # Official Kinetix v0.1.0 PPO defaults.
    num_envs: int = 2048
    rollout_steps: int = 64
    num_minibatches: int = 32
    update_epochs: int = 8
    learning_rate: float = 5e-5
    adam_epsilon: float = 1e-5
    gamma: float = 0.995
    gae_lambda: float = 0.9
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 1.0

    # Official spatial entity-transformer defaults.
    transformer_depth: int = 2
    transformer_qkv_width: int = 16
    transformer_encoder_width: int = 128
    transformer_heads: int = 8
    actor_critic_depth: int = 5
    actor_critic_width: int = 128

    @property
    def steps_per_update(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def updates_per_distribution(self) -> int:
        return self.timesteps_per_distribution // self.steps_per_update

    @property
    def heldout_updates(self) -> int:
        return self.heldout_timesteps // self.steps_per_update

    @property
    def consumed_timesteps_per_distribution(self) -> int:
        return self.updates_per_distribution * self.steps_per_update

    @property
    def max_diversity(self) -> int:
        return max(self.diversity_levels)

    def validate(self) -> None:
        if self.diversity_levels != tuple(sorted(set(self.diversity_levels))):
            raise ValueError("diversity levels must be unique and increasing")
        if self.max_diversity <= 0 or self.heldout_tasks <= 0:
            raise ValueError("task-pool sizes must be positive")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches")
        if self.transformer_encoder_width % self.transformer_heads:
            raise ValueError("transformer width must be divisible by heads")
        if self.updates_per_distribution <= 0 or self.heldout_updates <= 0:
            raise ValueError("each run must contain at least one PPO update")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kinetix_commit": KINETIX_COMMIT,
            "protocol": asdict(self),
        }

    def upstream_config(self) -> dict[str, Any]:
        """Return the flat config consumed by Kinetix v0.1.0 helpers."""

        return {
            "env_name": "Kinetix-Entity-MultiDiscrete-v1",
            "env_size_name": "s",
            "env_size_type": "predefined",
            "num_polygons": self.num_polygons,
            "num_circles": self.num_circles,
            "num_joints": self.num_joints,
            "num_thrusters": self.num_thrusters,
            "frame_skip": self.frame_skip,
            "num_motor_bindings": 4,
            "num_thruster_bindings": 2,
            "dense_reward_scale": self.dense_reward_scale,
            "num_train_envs": self.num_envs,
            "num_steps": self.rollout_steps,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            "lr": self.learning_rate,
            "max_grad_norm": self.max_gradient_norm,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_epsilon,
            "ent_coef": self.entropy_coefficient,
            "vf_coef": self.value_coefficient,
            "fc_layer_depth": self.actor_critic_depth,
            "fc_layer_width": self.actor_critic_width,
            "activation": "tanh",
            "recurrent_model": False,
            "transformer_depth": self.transformer_depth,
            "transformer_size": self.transformer_qkv_width,
            "transformer_encoder_size": self.transformer_encoder_width,
            "num_heads": self.transformer_heads,
            "full_attention_mask": False,
            "aggregate_mode": "dummy_and_mean",
            "static_env_params": {
                "num_motor_bindings": 4,
                "num_thruster_bindings": 2,
            },
        }


PROTOCOL = KinetixProtocol()
PROTOCOL.validate()
PROTOCOL_ID = hashlib.sha256(
    json.dumps(PROTOCOL.identity_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
