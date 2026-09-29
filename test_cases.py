"""Twenty public compatibility tests for CS 4341 Assignment 3.

The tests intentionally accept any reinforcement-learning algorithm, reward
scale, and integer state encoding. They verify the common interface that
training, Gradescope, and the class tournament require.
"""
from __future__ import annotations

import math
import random
import signal
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

try:
    from . import main
    from .reinforcement_learning import (
        BaseAgent, EpisodeConfig, choose_action, encode_observation, train_agent,
    )
    from .tag import Action, Tag, TagState
except ImportError:
    import main
    from reinforcement_learning import (
        BaseAgent, EpisodeConfig, choose_action, encode_observation, train_agent,
    )
    from tag import Action, Tag, TagState


PUBLIC_CONFIGURATIONS = (
    (434_301, 4, 4, 2, 24),
    (434_302, 5, 5, 3, 36),
    (434_303, 6, 6, 4, 48),
    (434_304, 7, 7, 5, 60),
    (434_305, 8, 8, 6, 72),
)
PUBLIC_TRAINING_SEED = 434_399
PUBLIC_TRAINING_UPDATES = 1_000
TRAINING_TIME_LIMIT_SECONDS = 30.0
MOVE_TIME_LIMIT_SECONDS = 1.0
_PLACEHOLDER_GROUP_NAMES = {
    "",
    "replace-with-your-group-name",
    "todo",
    "group name",
}


@contextmanager
def _time_limit(seconds: float, label: str) -> Iterator[None]:
    """Interrupt slow calls on Unix and check elapsed time on every platform."""
    def timeout_handler(signum: int, frame: object) -> None:
        pytest.fail(f"{label} reached its {seconds:g}-second limit")

    can_interrupt = hasattr(signal, "setitimer")
    if can_interrupt:
        previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.perf_counter()
    try:
        yield
    finally:
        if can_interrupt:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    elapsed = time.perf_counter() - started
    assert elapsed < seconds, (
        f"{label} must take less than {seconds:g} seconds; "
        f"this call took {elapsed:.3f} seconds"
    )


def _problem(configuration: tuple[int, int, int, int, int]) -> Tag:
    seed, n_rows, n_cols, n_players, max_turns = configuration
    players = tuple(f"P{index}" for index in range(n_players))
    return Tag.random(
        n_rows=n_rows,
        n_cols=n_cols,
        players=players,
        max_turns=max_turns,
        seed=seed,
    )


def _call_select(agent: BaseAgent, state: TagState) -> Action | None:
    with _time_limit(MOVE_TIME_LIMIT_SECONDS, "encode_state and select_action"):
        action = choose_action(agent, state, training=False)
    assert action is None or type(action) is Action, (
        "select_action must return a legal Action or None for a terminal state"
    )
    return action


@pytest.fixture
def agent() -> BaseAgent:
    with _time_limit(MOVE_TIME_LIMIT_SECONDS, "Agent constructor"):
        result = main.Agent(seed=PUBLIC_TRAINING_SEED)
    assert isinstance(result, BaseAgent), "Agent must inherit from BaseAgent"
    return result


@pytest.fixture
def episode_agent(agent: BaseAgent) -> BaseAgent:
    with _time_limit(MOVE_TIME_LIMIT_SECONDS, "init_episode"):
        config = agent.init_episode()
    assert isinstance(config, EpisodeConfig)
    return agent


@pytest.fixture(scope="session")
def trained_agent() -> BaseAgent:
    with _time_limit(TRAINING_TIME_LIMIT_SECONDS, "supplied training loop"):
        return train_agent(
            main.Agent(seed=PUBLIC_TRAINING_SEED),
            seed=PUBLIC_TRAINING_SEED,
            updates=PUBLIC_TRAINING_UPDATES,
        )


def test_init_episode_returns_a_valid_square_configuration(agent: BaseAgent) -> None:
    with _time_limit(MOVE_TIME_LIMIT_SECONDS, "init_episode"):
        config = agent.init_episode()
    assert isinstance(config, EpisodeConfig)
    assert type(config.side_length) is int and config.side_length > 0
    assert type(config.n_opponents) is int and config.n_opponents > 0
    assert type(config.max_steps) is int and config.max_steps > 0
    assert config.n_opponents + 1 <= config.side_length ** 2


def test_update_model_handles_regular_and_terminal_transitions(
    episode_agent: BaseAgent,
) -> None:
    agent = episode_agent
    state = _problem(PUBLIC_CONFIGURATIONS[0]).initial
    for terminal in (False, True):
        before = replace(state, turn=state.max_turns - 1) if terminal else state
        problem = Tag(before)
        action = problem.actions(before)[0]
        after = problem.result(before, action)
        while (
            not problem.is_terminal(after)
            and after.next_player != before.next_player
        ):
            after = problem.result(after, problem.actions(after)[0])
        player = before.next_player
        with _time_limit(MOVE_TIME_LIMIT_SECONDS, "encode_state"):
            encoded = encode_observation(agent, before, player)
            next_encoded = encode_observation(agent, after, player)
        with _time_limit(MOVE_TIME_LIMIT_SECONDS, "update_model"):
            result = agent.update_model(
                encoded, action, 1.0, next_encoded, problem.actions(after),
                player, problem.is_terminal(after),
            )
        assert result is None, "update_model must update self and return None"


def test_submission_module_is_named_main_py() -> None:
    assert Path(main.__file__).name == "main.py"


def test_group_name_is_set() -> None:
    group_name = getattr(main, "GROUP_NAME", None)
    assert isinstance(group_name, str), "GROUP_NAME must be a string"
    assert group_name.strip().lower() not in _PLACEHOLDER_GROUP_NAMES, (
        "replace GROUP_NAME with the name your group will use in the tournament"
    )


@pytest.mark.parametrize(
    "configuration",
    PUBLIC_CONFIGURATIONS[:4],
    ids=lambda value: f"seed_{value[0]}",
)
def test_state_encoder_returns_a_stable_integer(
    configuration: tuple[int, int, int, int, int],
    episode_agent: BaseAgent,
) -> None:
    agent = episode_agent
    state = _problem(configuration).initial
    player = state.next_player
    with _time_limit(MOVE_TIME_LIMIT_SECONDS, "encode_state"):
        first = encode_observation(agent, state, player)
        second = encode_observation(agent, state, player)
    assert type(first) is int, "encode_state must return a Python int"
    assert first == second, "encode_state must be deterministic"


@pytest.mark.parametrize(
    "configuration",
    PUBLIC_CONFIGURATIONS[:4],
    ids=lambda value: f"seed_{value[0]}",
)
def test_reward_returns_a_finite_float(
    configuration: tuple[int, int, int, int, int],
    episode_agent: BaseAgent,
) -> None:
    agent = episode_agent
    problem = _problem(configuration)
    state = problem.initial
    action = problem.actions(state)[0]
    next_state = problem.result(state, action)
    while (
        not problem.is_terminal(next_state)
        and next_state.next_player != state.next_player
    ):
        next_state = problem.result(next_state, problem.actions(next_state)[0])
    player = state.next_player
    with _time_limit(MOVE_TIME_LIMIT_SECONDS, "calculate_reward"):
        value = agent.calculate_reward(
            state, action, next_state, player, problem.is_terminal(next_state),
        )
    assert type(value) is float, (
        f"calculate_reward must return a Python float; got {type(value).__name__}"
    )
    assert math.isfinite(value), "calculate_reward must return a finite value"


def test_supplied_training_loop_completes(trained_agent: BaseAgent) -> None:
    assert isinstance(trained_agent, BaseAgent)


@pytest.mark.parametrize(
    "configuration",
    PUBLIC_CONFIGURATIONS,
    ids=lambda value: f"seed_{value[0]}",
)
def test_agent_returns_a_timely_legal_action(
    configuration: tuple[int, int, int, int, int],
    trained_agent: BaseAgent,
) -> None:
    problem = _problem(configuration)
    state = problem.initial
    action = _call_select(trained_agent, state)
    assert action in problem.actions(state), (
        "select_action must return an action from Tag.actions(state); "
        f"got {action!r}"
    )


def test_agent_returns_none_for_a_terminal_state(
    trained_agent: BaseAgent,
) -> None:
    state = _problem(PUBLIC_CONFIGURATIONS[0]).initial
    terminal = replace(state, turn=state.max_turns)
    action = _call_select(trained_agent, terminal)
    assert action is None, (
        "select_action must return None for a terminal state; "
        f"got {action!r}"
    )


def test_agent_completes_a_seeded_game(trained_agent: BaseAgent) -> None:
    problem = Tag.random(
        n_rows=5,
        n_cols=5,
        players=("A", "B", "C"),
        max_turns=30,
        seed=434_398,
    )
    state = problem.initial
    rng = random.Random(434_398)

    while not problem.is_terminal(state):
        # The trained policy controls every third turn; seeded random policies
        # supply the other actions.
        if state.next_player == "A":
            action = _call_select(trained_agent, state)
            assert action in problem.actions(state)
        else:
            action = rng.choice(problem.actions(state))
        state = problem.result(state, action)

    assert state.turn == state.max_turns
