from td2_kinetix.config import KINETIX_COMMIT, PROTOCOL, PROTOCOL_ID


def test_protocol_matches_requested_and_upstream_defaults():
    assert KINETIX_COMMIT == "eed109cb40d777f5ca62eddda134d74d92543870"
    assert PROTOCOL.learner_seeds == (0, 1, 2)
    assert PROTOCOL.diversity_levels == (1, 16, 64, 256, 65_536)
    assert PROTOCOL.num_distributions == 7
    assert PROTOCOL.timesteps_per_distribution == 100_000_000
    assert PROTOCOL.heldout_tasks == 100
    assert PROTOCOL.updates_per_distribution == 762
    assert PROTOCOL.consumed_timesteps_per_distribution == 99_876_864
    assert PROTOCOL.heldout_updates == 762
    assert len(PROTOCOL_ID) == 64


def test_upstream_config_is_entity_transformer_ppo():
    config = PROTOCOL.upstream_config()
    assert config["env_name"] == "Kinetix-Entity-MultiDiscrete-v1"
    assert config["env_size_name"] == "s"
    assert [config[k] for k in ("num_polygons", "num_circles", "num_joints", "num_thrusters")] == [5, 2, 1, 1]
    assert config["num_train_envs"] == 2048
    assert config["num_steps"] == 64
    assert config["transformer_depth"] == 2
    assert config["transformer_size"] == 16
    assert config["transformer_encoder_size"] == 128
    assert config["num_heads"] == 8
    assert config["fc_layer_depth"] == 5
    assert config["recurrent_model"] is False
    # EnvParams contributes another factor of 0.1, so passing 2.0 to the
    # wrapper implements the paper's effective dense shaping scale of 0.2.
    assert config["dense_reward_scale"] == 2.0
