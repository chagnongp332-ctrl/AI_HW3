"""Agent interface and supplied training loop for CS 4341 Assignment 3."""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from .tag import Action, Player, Tag, TagState
except ImportError:
    from tag import Action, Player, Tag, TagState


@dataclass(frozen=True)
class EpisodeConfig:
    """Square board, opponent count, and total player turns per episode."""

    side_length: int
    n_opponents: int
    max_steps: int

    def __post_init__(self) -> None:
        if type(self.side_length) is not int or self.side_length < 1:
            raise ValueError("side_length must be a positive integer.")
        if type(self.n_opponents) is not int or self.n_opponents < 1:
            raise ValueError("n_opponents must be a positive integer.")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        if self.n_opponents + 1 > self.side_length ** 2:
            raise ValueError("The learner plus opponents exceed the grid's spaces.")


class BaseAgent(ABC):
    """Implement five methods in main.Agent; keep learned data on self."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    @abstractmethod
    def init_episode(self) -> EpisodeConfig:
        """Choose the next episode's configuration and reset episode history."""
        raise NotImplementedError

    @abstractmethod
    def encode_state(self, state: TagState, player: Player) -> int:
        """Convert this player's view of the board to a deterministic Python int.

        Store encoded integers in your model and history; keep this conversion
        free of side effects. calculate_reward also receives the actual boards.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(
        self,
        state: TagState,
        action: Action,
        next_state: TagState,
        player: Player,
        terminal: bool,
    ) -> float:
        """Return a finite Python float using this player's actual board transition."""
        raise NotImplementedError

    @abstractmethod
    def select_action(
        self, state: int, legal_actions: tuple[Action, ...], training: bool,
    ) -> Action | None:
        """Choose a supplied legal action, using exploration when training=True.

        Return None when legal_actions is empty. Model updates belong in update_model.
        """
        raise NotImplementedError

    @abstractmethod
    def update_model(
        self,
        state: int,
        action: Action,
        reward: float,
        next_state: int,
        next_actions: tuple[Action, ...],
        player: Player,
        terminal: bool,
    ) -> None:
        """Learn from encoded states in place and return None.

        next_actions lists this player's next legal choices. It is empty when
        terminal is True, and that transition has zero future action value.
        """
        raise NotImplementedError


def encode_observation(agent: BaseAgent, state: TagState, player: Player) -> int:
    """Framework boundary: convert a board to a validated integer observation."""
    encoded = agent.encode_state(state, player)
    if type(encoded) is not int:
        raise TypeError("encode_state must return a Python int (not bool or a subclass).")
    return encoded


def _select_encoded_action(
    agent: BaseAgent,
    state: int,
    legal_actions: tuple[Action, ...],
    training: bool,
) -> Action | None:
    action = agent.select_action(state, legal_actions, training=training)
    if not legal_actions:
        if action is not None:
            raise ValueError("select_action must return None for a terminal state.")
    elif type(action) is not Action or action not in legal_actions:
        raise ValueError("select_action must return a legal Action from legal_actions.")
    return action


def choose_action(agent: BaseAgent, state: TagState, training: bool) -> Action | None:
    """Encode a board, call the student's policy, and validate its action."""
    encoded = encode_observation(agent, state, state.next_player)
    return _select_encoded_action(agent, encoded, Tag(state).actions(state), training)


def train_agent(agent: BaseAgent, seed: int, updates: int) -> BaseAgent:
    """Run self-play for exactly updates calls to update_model.

    One agent controls every player with training=True. The framework supplies
    integer encodings to policy and update methods, and actual boards to reward.
    Transitions span one player's action through its next turn or game end.
    Finished episodes restart while budget remains; the update budget can stop
    the final episode partway through its turns or terminal updates.
    """
    if not isinstance(agent, BaseAgent):
        raise TypeError("Agent must inherit from reinforcement_learning.BaseAgent.")
    if type(seed) is not int:
        raise TypeError("seed must be an integer.")
    if type(updates) is not int or updates < 0:
        raise ValueError("updates must be a nonnegative integer.")
    rng = random.Random(seed)
    completed = 0
    while completed < updates:
        config = agent.init_episode()
        if not isinstance(config, EpisodeConfig):
            raise TypeError("init_episode must return an EpisodeConfig.")
        players = tuple(f"P{index}" for index in range(config.n_opponents + 1))
        problem = Tag.random(
            n_rows=config.side_length,
            n_cols=config.side_length,
            players=players,
            max_turns=config.max_steps,
            seed=rng.randrange(2**63),
        )
        state = problem.initial
        pending: dict[Player, tuple[TagState, int, Action]] = {}

        def finish_transition(
            player: Player,
            next_state: TagState,
            next_encoded: int,
            next_actions: tuple[Action, ...],
            terminal: bool,
        ) -> None:
            previous, previous_encoded, action = pending.pop(player)
            value = agent.calculate_reward(
                previous, action, next_state, player, terminal,
            )
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("calculate_reward must return a finite Python float.")
            result = agent.update_model(
                previous_encoded, action, value, next_encoded,
                next_actions, player, terminal,
            )
            if result is not None:
                raise TypeError("update_model must update self and return None.")

        while not problem.is_terminal(state):
            player = state.next_player
            encoded = encode_observation(agent, state, player)
            legal_actions = problem.actions(state)
            if player in pending:
                finish_transition(player, state, encoded, legal_actions, terminal=False)
                completed += 1
                if completed == updates:
                    return agent
            action = _select_encoded_action(
                agent, encoded, legal_actions, training=True,
            )
            pending[player] = (state, encoded, action)
            state = problem.result(state, action)

        # A short episode may finish before every player gets a turn.
        for player in players:
            if player in pending:
                encoded = encode_observation(agent, state, player)
                finish_transition(player, state, encoded, (), terminal=True)
                completed += 1
                if completed == updates:
                    return agent
    return agent
