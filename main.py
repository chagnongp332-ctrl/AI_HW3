"""Complete the Agent class for CS 4341 Assignment 3."""
from __future__ import annotations

try:
    from .reinforcement_learning import BaseAgent, EpisodeConfig
    from .tag import Action, Player, TagState
except ImportError:
    from reinforcement_learning import BaseAgent, EpisodeConfig
    from tag import Action, Player, TagState


GROUP_NAME = "replace-with-your-group-name"


class Agent(BaseAgent):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        # Initialize your model and hyperparameters here. Use self.rng for
        # reproducible randomness. The supplied framework runs training.

    def init_episode(self) -> EpisodeConfig:
        """Return EpisodeConfig(side_length=..., n_opponents=..., max_steps=...).

        All three fields are positive integers. n_opponents + 1 must fit in
        side_length ** 2 cells. max_steps counts total player turns.
        Reset episode history here and preserve your learned model.
        """
        raise NotImplementedError

    def encode_state(self, state: TagState, player: Player) -> int:
        """Convert the board from player's perspective to a stable Python int.

        Include the features your policy and model need. Keep this conversion
        free of side effects and retain encoded integers in your model and history.
        """
        raise NotImplementedError

    def calculate_reward(
        self,
        state: TagState,
        action: Action,
        next_state: TagState,
        player: Player,
        terminal: bool,
    ) -> float:
        """Return a finite float using the actual before-and-after TagState objects.

        The interval spans player's action through its next turn or game end,
        including other players' moves. terminal says whether next_state ends
        the episode. Keep state objects local to this reward calculation.
        """
        raise NotImplementedError

    def select_action(
        self, state: int, legal_actions: tuple[Action, ...], training: bool,
    ) -> Action | None:
        """Choose from legal_actions using the encoded integer state.

        Use exploration when training=True and your evaluation policy when
        training=False. Return None when legal_actions is empty.
        """
        raise NotImplementedError

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
        """Update self from encoded states and the supplied reward; return None.

        next_actions contains this player's next legal choices. When terminal
        is True, it is empty and the transition has zero future action value.
        """
        raise NotImplementedError
