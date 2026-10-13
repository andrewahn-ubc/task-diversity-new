"""Typed, immutable configuration loading for Task Diversity 2.

The supplied source YAML intentionally contains scientific-notation scalars
(``1e8`` and ``4e7``) that PyYAML reads as strings.  Conversion is therefore
centralized here and validated before any value can reach a JAX scan length or
an Optax schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration is missing, inconsistent, or ambiguous."""


@dataclass(frozen=True, slots=True)
class RulesetConfig:
    env_name: str
    tree_depth: int
    ruleset_base_seed: int
    use_ruleset_dataset: bool
    ruleset_dataset_file: str
    auto_generate_dataset_if_missing: bool
    dataset_gen_chunk: int
    dataset_generation_python_exec: str | None
    dataset_generation_dry_run: bool
    use_dataset_distractors: bool
    distractor_combine_penalty: float
    balanced_depth_sampling: bool
    env_structure_sample: bool
    env_structure_prob: float
    env_structure_types: tuple[str, ...] | None
    slow_tile_prob: float
    include_rules_in_obs: bool
    ruleset_embed_dim: int
    ruleset_embed_mode: str


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float
    anneal_learning_rate: bool
    learning_rate_warmup_fraction: float
    num_envs: int
    num_steps: int
    max_steps: int
    update_epochs: int
    num_minibatches: int
    total_timesteps: int
    reward_shaping_horizon: int
    actor_clip_epsilon: float
    value_clip_epsilon: float
    entropy_coefficient: float
    gamma: float
    gae_lambda: float
    scale_clip_epsilon: bool
    value_coefficient: float
    max_gradient_norm: float


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    fc_dim: int
    recurrent_hidden_dim: int
    recurrent_type: str
    activation: str
    embedding_type: str
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """Hidden FC depth; logits and value projections are not counted."""

    shared_pre_gru_fc_layers: int
    actor_head_fc_layers: int
    critic_head_fc_layers: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int
    num_seeds: int
    entity: str
    project: str
    wandb_mode: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed, self.seed + self.num_seeds))


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    timing_chunk_size: int
    disable_wandb: bool
    disable_visualization: bool
    disable_evaluation: bool
    disable_async_evaluation: bool
    disable_all_callbacks: bool
    skip_step_diagnostics: bool


@dataclass(frozen=True, slots=True)
class AdaptationConfig:
    enabled: bool
    after_distributions: tuple[int, ...]
    additional_timesteps: int
    heldout_tasks: int
    heldout_task_seed: int
    fixed_across_distributions: bool
    fixed_across_diversity_streams: bool
    require_disjoint_from_training: bool
    discard_branch_updates: bool


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    num_agents: int
    num_distributions: int
    diversity_levels: tuple[int, ...]
    methods: tuple[str, ...]
    ambiguity_resolution: str
    adaptation: AdaptationConfig


@dataclass(frozen=True, slots=True)
class RunSchedule:
    """Derived work for one training distribution or adaptation branch."""

    requested_timesteps: int
    environment_steps_per_update: int
    num_updates: int
    consumed_timesteps: int
    timestep_remainder: int
    num_actors: int
    actor_transitions_per_update: int
    actor_sequences_per_minibatch: int
    actor_transitions_per_minibatch: int
    optimizer_steps_per_update: int
    total_optimizer_steps: int
    warmup_optimizer_steps: int


@dataclass(frozen=True, slots=True)
class DerivedConfig:
    training_per_distribution: RunSchedule
    adaptation_per_probe: RunSchedule | None
    total_training_updates_per_stream: int
    total_training_optimizer_steps_per_stream: int
    total_training_warmup_optimizer_steps_per_stream: int
    adaptation_probe_count: int


@dataclass(frozen=True, slots=True)
class TD2Config:
    """Fully normalized configuration used by experiment code."""

    ruleset: RulesetConfig
    ppo: PPOConfig
    network: NetworkConfig
    architecture: ArchitectureConfig
    runtime: RuntimeConfig
    profiling: ProfilingConfig
    experiment: ExperimentConfig
    derived: DerivedConfig
    source_files: tuple[Path, ...]


_SOURCE_KEYS = frozenset(
    {
        "ENV_NAME",
        "TREE_DEPTH",
        "RULESET_BASE_SEED",
        "USE_RULESET_DATASET",
        "RULESET_DATASET_FILE",
        "AUTO_GENERATE_DATASET_IF_MISSING",
        "DATASET_GEN_CHUNK",
        "DATASET_GENERATION_PYTHON_EXEC",
        "DATASET_GENERATION_DRY_RUN",
        "USE_DATASET_DISTRACTORS",
        "DISTRACTOR_COMBINE_PENALTY",
        "BALANCED_DEPTH_SAMPLING",
        "ENV_STRUCTURE_SAMPLE",
        "ENV_STRUCTURE_PROB",
        "ENV_STRUCTURE_TYPES",
        "SLOW_TILE_PROB",
        "INCLUDE_RULES_IN_OBS",
        "RULESET_EMBED_DIM",
        "RULESET_EMBED_MODE",
        "LR",
        "ANNEAL_LR",
        "LR_WARMUP",
        "NUM_ENVS",
        "NUM_STEPS",
        "MAX_STEPS",
        "UPDATE_EPOCHS",
        "NUM_MINIBATCHES",
        "TOTAL_TIMESTEPS",
        "REW_SHAPING_HORIZON",
        "FC_DIM_SIZE",
        "GRU_HIDDEN_DIM",
        "RNN_TYPE",
        "ACTOR_CLIP_EPS",
        "VALUE_CLIP_EPS",
        "ENT_COEF",
        "GAMMA",
        "GAE_LAMBDA",
        "SCALE_CLIP_EPS",
        "VF_COEF",
        "MAX_GRAD_NORM",
        "ACTIVATION",
        "SEED",
        "NUM_SEEDS",
        "EMBEDDING",
        "EMBED_DIM",
        "ENTITY",
        "PROJECT",
        "WANDB_MODE",
        "TIMING_CHUNK_SIZE",
        "DISABLE_WANDB",
        "DISABLE_VIZ",
        "DISABLE_EVAL",
        "DISABLE_ASYNC_EVAL",
        "DISABLE_ALL_CALLBACKS",
        "SKIP_STEP_DIAGNOSTICS",
    }
)

_TOP_LEVEL_SECTIONS = frozenset(
    {"architecture", "experiment", "derived_expectations"}
)


def load_config(path: str | Path) -> TD2Config:
    """Load an inherited YAML configuration into an immutable model.

    ``extends`` paths are resolved relative to the file that contains them.
    Mappings are merged recursively; sequences and scalar values are replaced.
    """

    final_path = Path(path).expanduser().resolve()
    data, source_files = _load_inherited_yaml(final_path, ())
    return _build_config(data, source_files)


def _load_inherited_yaml(
    path: Path, stack: tuple[Path, ...]
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigurationError(f"configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from error

    if not isinstance(loaded, dict):
        raise ConfigurationError(f"top level of {path} must be a mapping")
    local = dict(loaded)
    parent_ref = local.pop("extends", None)
    if parent_ref is None:
        return local, (path,)
    if not isinstance(parent_ref, str) or not parent_ref.strip():
        raise ConfigurationError(f"{path}: extends must be a non-empty path string")

    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent_path = parent_path.resolve()
    parent, parent_sources = _load_inherited_yaml(parent_path, (*stack, path))
    return _deep_merge(parent, local), (*parent_sources, path)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(previous, value)
        else:
            merged[key] = value
    return merged


def _build_config(data: Mapping[str, Any], source_files: tuple[Path, ...]) -> TD2Config:
    _expect_keys(
        data,
        required=_SOURCE_KEYS | _TOP_LEVEL_SECTIONS,
        optional=frozenset(),
        scope="configuration",
    )

    ruleset = RulesetConfig(
        env_name=_as_str(data["ENV_NAME"], "ENV_NAME"),
        tree_depth=_positive_int(data["TREE_DEPTH"], "TREE_DEPTH"),
        ruleset_base_seed=_as_int(data["RULESET_BASE_SEED"], "RULESET_BASE_SEED"),
        use_ruleset_dataset=_as_bool(data["USE_RULESET_DATASET"], "USE_RULESET_DATASET"),
        ruleset_dataset_file=_as_str(
            data["RULESET_DATASET_FILE"], "RULESET_DATASET_FILE"
        ),
        auto_generate_dataset_if_missing=_as_bool(
            data["AUTO_GENERATE_DATASET_IF_MISSING"],
            "AUTO_GENERATE_DATASET_IF_MISSING",
        ),
        dataset_gen_chunk=_positive_int(data["DATASET_GEN_CHUNK"], "DATASET_GEN_CHUNK"),
        dataset_generation_python_exec=_optional_str(
            data["DATASET_GENERATION_PYTHON_EXEC"],
            "DATASET_GENERATION_PYTHON_EXEC",
        ),
        dataset_generation_dry_run=_as_bool(
            data["DATASET_GENERATION_DRY_RUN"], "DATASET_GENERATION_DRY_RUN"
        ),
        use_dataset_distractors=_as_bool(
            data["USE_DATASET_DISTRACTORS"], "USE_DATASET_DISTRACTORS"
        ),
        distractor_combine_penalty=_as_float(
            data["DISTRACTOR_COMBINE_PENALTY"], "DISTRACTOR_COMBINE_PENALTY"
        ),
        balanced_depth_sampling=_as_bool(
            data["BALANCED_DEPTH_SAMPLING"], "BALANCED_DEPTH_SAMPLING"
        ),
        env_structure_sample=_as_bool(
            data["ENV_STRUCTURE_SAMPLE"], "ENV_STRUCTURE_SAMPLE"
        ),
        env_structure_prob=_probability(
            data["ENV_STRUCTURE_PROB"], "ENV_STRUCTURE_PROB"
        ),
        env_structure_types=_optional_str_tuple(
            data["ENV_STRUCTURE_TYPES"], "ENV_STRUCTURE_TYPES"
        ),
        slow_tile_prob=_probability(data["SLOW_TILE_PROB"], "SLOW_TILE_PROB"),
        include_rules_in_obs=_as_bool(
            data["INCLUDE_RULES_IN_OBS"], "INCLUDE_RULES_IN_OBS"
        ),
        ruleset_embed_dim=_positive_int(
            data["RULESET_EMBED_DIM"], "RULESET_EMBED_DIM"
        ),
        ruleset_embed_mode=_choice(
            data["RULESET_EMBED_MODE"],
            "RULESET_EMBED_MODE",
            {"concat", "replace", "structured"},
        ),
    )

    ppo = PPOConfig(
        learning_rate=_positive_float(data["LR"], "LR"),
        anneal_learning_rate=_as_bool(data["ANNEAL_LR"], "ANNEAL_LR"),
        learning_rate_warmup_fraction=_unit_interval(
            data["LR_WARMUP"], "LR_WARMUP"
        ),
        num_envs=_positive_int(data["NUM_ENVS"], "NUM_ENVS"),
        num_steps=_positive_int(data["NUM_STEPS"], "NUM_STEPS"),
        max_steps=_positive_int(data["MAX_STEPS"], "MAX_STEPS"),
        update_epochs=_positive_int(data["UPDATE_EPOCHS"], "UPDATE_EPOCHS"),
        num_minibatches=_positive_int(
            data["NUM_MINIBATCHES"], "NUM_MINIBATCHES"
        ),
        total_timesteps=_positive_int(data["TOTAL_TIMESTEPS"], "TOTAL_TIMESTEPS"),
        reward_shaping_horizon=_nonnegative_int(
            data["REW_SHAPING_HORIZON"], "REW_SHAPING_HORIZON"
        ),
        actor_clip_epsilon=_positive_float(
            data["ACTOR_CLIP_EPS"], "ACTOR_CLIP_EPS"
        ),
        value_clip_epsilon=_positive_float(
            data["VALUE_CLIP_EPS"], "VALUE_CLIP_EPS"
        ),
        entropy_coefficient=_nonnegative_float(data["ENT_COEF"], "ENT_COEF"),
        gamma=_unit_interval(data["GAMMA"], "GAMMA"),
        gae_lambda=_unit_interval(data["GAE_LAMBDA"], "GAE_LAMBDA"),
        scale_clip_epsilon=_as_bool(data["SCALE_CLIP_EPS"], "SCALE_CLIP_EPS"),
        value_coefficient=_nonnegative_float(data["VF_COEF"], "VF_COEF"),
        max_gradient_norm=_positive_float(
            data["MAX_GRAD_NORM"], "MAX_GRAD_NORM"
        ),
    )
    if ppo.reward_shaping_horizon > ppo.total_timesteps:
        raise ConfigurationError(
            "REW_SHAPING_HORIZON cannot exceed TOTAL_TIMESTEPS"
        )

    network = NetworkConfig(
        fc_dim=_positive_int(data["FC_DIM_SIZE"], "FC_DIM_SIZE"),
        recurrent_hidden_dim=_positive_int(
            data["GRU_HIDDEN_DIM"], "GRU_HIDDEN_DIM"
        ),
        recurrent_type=_choice(data["RNN_TYPE"], "RNN_TYPE", {"GRU", "LSTM"}),
        activation=_choice(data["ACTIVATION"], "ACTIVATION", {"relu", "tanh"}),
        embedding_type=_choice(data["EMBEDDING"], "EMBEDDING", {"float"}),
        embedding_dim=_positive_int(data["EMBED_DIM"], "EMBED_DIM"),
    )

    architecture_data = _as_mapping(data["architecture"], "architecture")
    _expect_keys(
        architecture_data,
        required=frozenset(
            {
                "shared_pre_gru_fc_layers",
                "actor_head_fc_layers",
                "critic_head_fc_layers",
            }
        ),
        optional=frozenset(),
        scope="architecture",
    )
    architecture = ArchitectureConfig(
        shared_pre_gru_fc_layers=_positive_int(
            architecture_data["shared_pre_gru_fc_layers"],
            "architecture.shared_pre_gru_fc_layers",
        ),
        actor_head_fc_layers=_positive_int(
            architecture_data["actor_head_fc_layers"],
            "architecture.actor_head_fc_layers",
        ),
        critic_head_fc_layers=_positive_int(
            architecture_data["critic_head_fc_layers"],
            "architecture.critic_head_fc_layers",
        ),
    )
    expected_depth = ArchitectureConfig(2, 2, 2)
    if architecture != expected_depth:
        raise ConfigurationError(
            "this implementation matches the supplied authors' architecture at "
            "two shared pre-GRU FCs and two FCs in each actor/critic head"
        )

    runtime = RuntimeConfig(
        seed=_as_int(data["SEED"], "SEED"),
        num_seeds=_positive_int(data["NUM_SEEDS"], "NUM_SEEDS"),
        entity=_as_str(data["ENTITY"], "ENTITY", allow_empty=True),
        project=_as_str(data["PROJECT"], "PROJECT", allow_empty=True),
        wandb_mode=_choice(
            data["WANDB_MODE"], "WANDB_MODE", {"online", "offline", "disabled"}
        ),
    )

    profiling = ProfilingConfig(
        timing_chunk_size=_positive_int(
            data["TIMING_CHUNK_SIZE"], "TIMING_CHUNK_SIZE"
        ),
        disable_wandb=_as_bool(data["DISABLE_WANDB"], "DISABLE_WANDB"),
        disable_visualization=_as_bool(data["DISABLE_VIZ"], "DISABLE_VIZ"),
        disable_evaluation=_as_bool(data["DISABLE_EVAL"], "DISABLE_EVAL"),
        disable_async_evaluation=_as_bool(
            data["DISABLE_ASYNC_EVAL"], "DISABLE_ASYNC_EVAL"
        ),
        disable_all_callbacks=_as_bool(
            data["DISABLE_ALL_CALLBACKS"], "DISABLE_ALL_CALLBACKS"
        ),
        skip_step_diagnostics=_as_bool(
            data["SKIP_STEP_DIAGNOSTICS"], "SKIP_STEP_DIAGNOSTICS"
        ),
    )

    experiment_data = _as_mapping(data["experiment"], "experiment")
    _expect_keys(
        experiment_data,
        required=frozenset(
            {
                "num_agents",
                "num_distributions",
                "diversity_levels",
                "methods",
                "ambiguity_resolution",
                "adaptation",
            }
        ),
        optional=frozenset(),
        scope="experiment",
    )
    adaptation_data = _as_mapping(
        experiment_data["adaptation"], "experiment.adaptation"
    )
    _expect_keys(
        adaptation_data,
        required=frozenset(
            {
                "enabled",
                "after_distributions",
                "additional_timesteps",
                "heldout_tasks",
                "heldout_task_seed",
                "fixed_across_distributions",
                "fixed_across_diversity_streams",
                "require_disjoint_from_training",
                "discard_branch_updates",
            }
        ),
        optional=frozenset(),
        scope="experiment.adaptation",
    )
    adaptation = AdaptationConfig(
        enabled=_as_bool(adaptation_data["enabled"], "experiment.adaptation.enabled"),
        after_distributions=_int_tuple(
            adaptation_data["after_distributions"],
            "experiment.adaptation.after_distributions",
        ),
        additional_timesteps=_positive_int(
            adaptation_data["additional_timesteps"],
            "experiment.adaptation.additional_timesteps",
        ),
        heldout_tasks=_positive_int(
            adaptation_data["heldout_tasks"], "experiment.adaptation.heldout_tasks"
        ),
        heldout_task_seed=_as_int(
            adaptation_data["heldout_task_seed"],
            "experiment.adaptation.heldout_task_seed",
        ),
        fixed_across_distributions=_as_bool(
            adaptation_data["fixed_across_distributions"],
            "experiment.adaptation.fixed_across_distributions",
        ),
        fixed_across_diversity_streams=_as_bool(
            adaptation_data["fixed_across_diversity_streams"],
            "experiment.adaptation.fixed_across_diversity_streams",
        ),
        require_disjoint_from_training=_as_bool(
            adaptation_data["require_disjoint_from_training"],
            "experiment.adaptation.require_disjoint_from_training",
        ),
        discard_branch_updates=_as_bool(
            adaptation_data["discard_branch_updates"],
            "experiment.adaptation.discard_branch_updates",
        ),
    )
    experiment = ExperimentConfig(
        num_agents=_positive_int(
            experiment_data["num_agents"], "experiment.num_agents"
        ),
        num_distributions=_positive_int(
            experiment_data["num_distributions"], "experiment.num_distributions"
        ),
        diversity_levels=_positive_int_tuple(
            experiment_data["diversity_levels"], "experiment.diversity_levels"
        ),
        methods=_str_tuple(experiment_data["methods"], "experiment.methods"),
        ambiguity_resolution=_as_str(
            experiment_data["ambiguity_resolution"],
            "experiment.ambiguity_resolution",
        ),
        adaptation=adaptation,
    )
    _validate_experiment(experiment)

    training_schedule = _derive_schedule(
        requested_timesteps=ppo.total_timesteps,
        ppo=ppo,
        num_agents=experiment.num_agents,
        scope="training distribution",
    )
    adaptation_schedule = (
        _derive_schedule(
            requested_timesteps=adaptation.additional_timesteps,
            ppo=ppo,
            num_agents=experiment.num_agents,
            scope="adaptation probe",
        )
        if adaptation.enabled
        else None
    )
    derived = DerivedConfig(
        training_per_distribution=training_schedule,
        adaptation_per_probe=adaptation_schedule,
        total_training_updates_per_stream=(
            training_schedule.num_updates * experiment.num_distributions
        ),
        total_training_optimizer_steps_per_stream=(
            training_schedule.total_optimizer_steps * experiment.num_distributions
        ),
        total_training_warmup_optimizer_steps_per_stream=(
            int(
                training_schedule.num_updates
                * experiment.num_distributions
                * ppo.learning_rate_warmup_fraction
            )
            * training_schedule.optimizer_steps_per_update
        ),
        adaptation_probe_count=len(adaptation.after_distributions)
        if adaptation.enabled
        else 0,
    )
    expectations = _as_mapping(
        data["derived_expectations"], "derived_expectations"
    )
    _validate_derived_expectations(expectations, derived)

    return TD2Config(
        ruleset=ruleset,
        ppo=ppo,
        network=network,
        architecture=architecture,
        runtime=runtime,
        profiling=profiling,
        experiment=experiment,
        derived=derived,
        source_files=source_files,
    )


def _derive_schedule(
    *, requested_timesteps: int, ppo: PPOConfig, num_agents: int, scope: str
) -> RunSchedule:
    environment_steps_per_update = ppo.num_envs * ppo.num_steps
    num_updates, remainder = divmod(requested_timesteps, environment_steps_per_update)
    if num_updates < 1:
        raise ConfigurationError(
            f"{scope} has {requested_timesteps} timesteps but one PPO update needs "
            f"{environment_steps_per_update}"
        )

    num_actors = ppo.num_envs * num_agents
    if num_actors % ppo.num_minibatches:
        raise ConfigurationError(
            f"{scope}: NUM_ENVS * num_agents ({num_actors}) must be divisible by "
            f"NUM_MINIBATCHES ({ppo.num_minibatches}) for recurrent sequence batching"
        )
    actor_transitions_per_update = num_actors * ppo.num_steps
    if actor_transitions_per_update % ppo.num_minibatches:
        raise ConfigurationError(
            f"{scope}: actor transitions per update must divide evenly into minibatches"
        )

    optimizer_steps_per_update = ppo.update_epochs * ppo.num_minibatches
    total_optimizer_steps = num_updates * optimizer_steps_per_update
    # Match the authors' schedule exactly: truncate the warmup in PPO-update
    # units first, then convert that boundary to optimizer/minibatch steps.
    warmup_optimizer_steps = (
        int(num_updates * ppo.learning_rate_warmup_fraction)
        * optimizer_steps_per_update
    )
    return RunSchedule(
        requested_timesteps=requested_timesteps,
        environment_steps_per_update=environment_steps_per_update,
        num_updates=num_updates,
        consumed_timesteps=num_updates * environment_steps_per_update,
        timestep_remainder=remainder,
        num_actors=num_actors,
        actor_transitions_per_update=actor_transitions_per_update,
        actor_sequences_per_minibatch=num_actors // ppo.num_minibatches,
        actor_transitions_per_minibatch=(
            actor_transitions_per_update // ppo.num_minibatches
        ),
        optimizer_steps_per_update=optimizer_steps_per_update,
        total_optimizer_steps=total_optimizer_steps,
        warmup_optimizer_steps=warmup_optimizer_steps,
    )


def _validate_experiment(config: ExperimentConfig) -> None:
    if len(set(config.diversity_levels)) != len(config.diversity_levels):
        raise ConfigurationError("experiment.diversity_levels must be unique")
    if len(set(config.methods)) != len(config.methods):
        raise ConfigurationError("experiment.methods must be unique")

    adaptation = config.adaptation
    if not adaptation.enabled:
        if adaptation.after_distributions:
            raise ConfigurationError(
                "disabled adaptation must have an empty after_distributions list"
            )
        return

    expected_probes = tuple(range(1, config.num_distributions + 1))
    if adaptation.after_distributions != expected_probes:
        raise ConfigurationError(
            "adaptation probes must run after every distribution; expected "
            f"{expected_probes}, got {adaptation.after_distributions}"
        )
    if not adaptation.fixed_across_distributions:
        raise ConfigurationError("heldout tasks must be fixed across distributions")
    if not adaptation.fixed_across_diversity_streams:
        raise ConfigurationError("heldout tasks must be fixed across diversity streams")
    if not adaptation.require_disjoint_from_training:
        raise ConfigurationError("heldout tasks must be disjoint from training tasks")
    if not adaptation.discard_branch_updates:
        raise ConfigurationError(
            "adaptation updates must be discarded to preserve continual trajectories"
        )


def _validate_derived_expectations(
    expectations: Mapping[str, Any], derived: DerivedConfig
) -> None:
    required = frozenset(
        {
            "updates_per_distribution",
            "optimizer_steps_per_distribution",
            "warmup_optimizer_steps_per_distribution",
            "consumed_timesteps_per_distribution",
            "timestep_remainder_per_distribution",
            "total_training_updates_per_stream",
            "total_training_optimizer_steps_per_stream",
            "total_training_warmup_optimizer_steps_per_stream",
            "adaptation_probe_count",
            "updates_per_adaptation_probe",
            "optimizer_steps_per_adaptation_probe",
            "warmup_optimizer_steps_per_adaptation_probe",
        }
    )
    _expect_keys(
        expectations,
        required=required,
        optional=frozenset(),
        scope="derived_expectations",
    )
    adaptation = derived.adaptation_per_probe
    actual = {
        "updates_per_distribution": derived.training_per_distribution.num_updates,
        "optimizer_steps_per_distribution": (
            derived.training_per_distribution.total_optimizer_steps
        ),
        "warmup_optimizer_steps_per_distribution": (
            derived.training_per_distribution.warmup_optimizer_steps
        ),
        "consumed_timesteps_per_distribution": (
            derived.training_per_distribution.consumed_timesteps
        ),
        "timestep_remainder_per_distribution": (
            derived.training_per_distribution.timestep_remainder
        ),
        "total_training_updates_per_stream": derived.total_training_updates_per_stream,
        "total_training_optimizer_steps_per_stream": (
            derived.total_training_optimizer_steps_per_stream
        ),
        "total_training_warmup_optimizer_steps_per_stream": (
            derived.total_training_warmup_optimizer_steps_per_stream
        ),
        "adaptation_probe_count": derived.adaptation_probe_count,
        "updates_per_adaptation_probe": adaptation.num_updates if adaptation else 0,
        "optimizer_steps_per_adaptation_probe": (
            adaptation.total_optimizer_steps if adaptation else 0
        ),
        "warmup_optimizer_steps_per_adaptation_probe": (
            adaptation.warmup_optimizer_steps if adaptation else 0
        ),
    }
    for key, actual_value in actual.items():
        expected_value = _nonnegative_int(
            expectations[key], f"derived_expectations.{key}"
        )
        if expected_value != actual_value:
            raise ConfigurationError(
                f"derived_expectations.{key} expected {expected_value}, "
                f"but normalized configuration derives {actual_value}"
            )


def _expect_keys(
    mapping: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    scope: str,
) -> None:
    keys = frozenset(mapping)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ConfigurationError(f"{scope} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"{scope} has unknown keys: {', '.join(unknown)}")


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} keys must be strings")
    return value


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value) or not value.is_integer():
            raise ConfigurationError(f"{name} must be a finite integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation as error:
            raise ConfigurationError(
                f"{name} must be an integer or scientific-notation integer, got {value!r}"
            ) from error
        if not number.is_finite() or number != number.to_integral_value():
            raise ConfigurationError(f"{name} must be an integer, got {value!r}")
        return int(number)
    raise ConfigurationError(f"{name} must be an integer, got {type(value).__name__}")


def _positive_int(value: Any, name: str) -> int:
    result = _as_int(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive, got {result}")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = _as_int(value, name)
    if result < 0:
        raise ConfigurationError(f"{name} must be nonnegative, got {result}")
    return result


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be numeric, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be numeric, got {value!r}") from error
    if not isfinite(result):
        raise ConfigurationError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _as_float(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive, got {result}")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _as_float(value, name)
    if result < 0:
        raise ConfigurationError(f"{name} must be nonnegative, got {result}")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _as_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1], got {result}")
    return result


def _probability(value: Any, name: str) -> float:
    return _unit_interval(value, name)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a YAML boolean, got {value!r}")
    return value


def _as_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string, got {type(value).__name__}")
    result = value.strip()
    if not result and not allow_empty:
        raise ConfigurationError(f"{name} must not be empty")
    return result


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, name)


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _as_str(value, name)
    if result not in choices:
        rendered = ", ".join(sorted(choices))
        raise ConfigurationError(f"{name} must be one of {{{rendered}}}, got {result!r}")
    return result


def _sequence(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{name} must be a YAML sequence")
    return value


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    return tuple(_as_int(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name)))


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    result = tuple(
        _positive_int(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    )
    if not result:
        raise ConfigurationError(f"{name} must not be empty")
    return result


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    result = tuple(
        _as_str(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    )
    if not result:
        raise ConfigurationError(f"{name} must not be empty")
    return result


def _optional_str_tuple(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(
        _as_str(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    )
