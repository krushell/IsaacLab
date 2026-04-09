# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from collections import namedtuple

import pytest
import torch

from isaaclab.managers import RewardManager, RewardTermCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


def grilled_chicken(env):
    return 1


def grilled_chicken_with_bbq(env, bbq: bool):
    return 0


def grilled_chicken_with_curry(env, hot: bool):
    return 0


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return 0


@pytest.fixture
def env():
    sim = SimulationContext()
    return namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "sim"])(20, 0.1, "cpu", sim)


def test_str(env):
    """Test the string representation of the reward manager."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(
            func=grilled_chicken_with_yoghurt,
            weight=1.0,
            params={"hot": False, "bland": 2.0},
        ),
    }
    rew_man = RewardManager(cfg, env)
    assert len(rew_man.active_terms) == 3
    # print the expected string
    print()
    print(rew_man)


def test_config_equivalence(env):
    """Test the equivalence of reward manager created from different config types."""
    # create from dictionary
    cfg = {
        "my_term": RewardTermCfg(func=grilled_chicken, weight=10),
        "your_term": RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True}),
        "his_term": RewardTermCfg(
            func=grilled_chicken_with_yoghurt,
            weight=1.0,
            params={"hot": False, "bland": 2.0},
        ),
    }
    rew_man_from_dict = RewardManager(cfg, env)

    # create from config class
    @configclass
    class MyRewardManagerCfg:
        """Reward manager config with no type annotations."""

        my_term = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term = RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0})

    cfg = MyRewardManagerCfg()
    rew_man_from_cfg = RewardManager(cfg, env)

    # create from config class
    @configclass
    class MyRewardManagerAnnotatedCfg:
        """Reward manager config with type annotations."""

        my_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term: RewardTermCfg = RewardTermCfg(
            func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0}
        )

    cfg = MyRewardManagerAnnotatedCfg()
    rew_man_from_annotated_cfg = RewardManager(cfg, env)

    # check equivalence
    # parsed terms
    assert rew_man_from_dict.active_terms == rew_man_from_annotated_cfg.active_terms
    assert rew_man_from_cfg.active_terms == rew_man_from_annotated_cfg.active_terms
    assert rew_man_from_dict.active_terms == rew_man_from_cfg.active_terms
    # parsed term configs
    for term_name in ["my_term", "your_term", "his_term"]:
        assert rew_man_from_dict.get_term_cfg(term_name) == rew_man_from_annotated_cfg.get_term_cfg(term_name)
        assert rew_man_from_cfg.get_term_cfg(term_name) == rew_man_from_annotated_cfg.get_term_cfg(term_name)
        assert rew_man_from_dict.get_term_cfg(term_name) == rew_man_from_cfg.get_term_cfg(term_name)


def test_compute(env):
    """Test the computation of reward."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
    }
    rew_man = RewardManager(cfg, env)
    # compute expected reward
    expected_reward = cfg["term_1"].weight * env.dt
    # compute reward using manager
    rewards = rew_man.compute(dt=env.dt)
    # check the reward for environment index 0
    assert float(rewards[0]) == expected_reward
    assert tuple(rewards.shape) == (env.num_envs,)


def test_config_empty(env):
    """Test the creation of reward manager with empty config."""
    rew_man = RewardManager(None, env)
    assert len(rew_man.active_terms) == 0

    # print the expected string
    print()
    print(rew_man)

    # compute reward
    rewards = rew_man.compute(dt=env.dt)

    # check all rewards are zero
    torch.testing.assert_close(rewards, torch.zeros_like(rewards))


def test_active_terms(env):
    """Test the correct reading of active terms."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
    }
    rew_man = RewardManager(cfg, env)

    assert len(rew_man.active_terms) == 3


def test_get_term_cfg_no_group(env):
    """Test term lookup for ungrouped rewards."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken, weight=2),
    }
    rew_man = RewardManager(cfg, env)

    assert rew_man.get_term_cfg("term_1") == cfg["term_1"]

    with pytest.raises(ValueError, match="invalid for ungrouped rewards"):
        rew_man.get_term_cfg("reward/term_1")


def test_get_term_cfg_grouped(env):
    """Test term lookup for grouped rewards."""
    cfg = {
        "group_a": {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken, weight=2),
        },
        "group_b": {
            "term_3": RewardTermCfg(func=grilled_chicken, weight=3),
        },
    }
    rew_man = RewardManager(cfg, env)

    assert rew_man.get_term_cfg("group_a/term_1") == cfg["group_a"]["term_1"]

    with pytest.raises(ValueError, match="ambiguous for grouped rewards"):
        rew_man.get_term_cfg("term_1")
    with pytest.raises(ValueError, match="Reward group 'missing' not found."):
        rew_man.get_term_cfg("missing/term_1")


def test_get_active_iterable_terms_no_group(env):
    """Test active iterable terms for ungrouped rewards."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken, weight=2),
    }
    rew_man = RewardManager(cfg, env)

    rew_man.compute(dt=env.dt)

    assert rew_man.get_active_iterable_terms(0) == [
        ("term_1", [10.0]),
        ("term_2", [2.0]),
    ]


def test_get_active_iterable_terms_grouped(env):
    """Test active iterable terms for grouped rewards."""
    cfg = {
        "group_a": {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken, weight=2),
        },
        "group_b": {
            "term_3": RewardTermCfg(func=grilled_chicken, weight=3),
        },
    }
    rew_man = RewardManager(cfg, env)

    rew_man.compute(dt=env.dt)

    assert rew_man.get_active_iterable_terms(0) == [
        ("group_a-term_1", [10.0]),
        ("group_a-term_2", [2.0]),
        ("group_b-term_3", [3.0]),
    ]


def test_missing_weight(env):
    """Test the missing of weight in the config."""
    # TODO: The error should be raised during the config parsing, not during the reward manager creation.
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, params={"bbq": True}),
    }
    with pytest.raises(TypeError):
        RewardManager(cfg, env)


def test_invalid_reward_func_module(env):
    """Test the handling of invalid reward function's module in string representation."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func="a:grilled_chicken_with_no_bbq", weight=0.1, params={"hot": False}),
    }
    with pytest.raises(ValueError):
        RewardManager(cfg, env)


def test_invalid_reward_config(env):
    """Test the handling of invalid reward function's config parameters."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken_with_bbq, weight=0.1, params={"hot": False}),
        "term_2": RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=2.0, params={"hot": False}),
    }
    with pytest.raises(ValueError):
        RewardManager(cfg, env)
