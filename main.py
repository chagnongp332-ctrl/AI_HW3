"""Complete the Agent class for CS 4341 Assignment 3."""
from __future__ import annotations

try:
    from .reinforcement_learning import BaseAgent, EpisodeConfig
    from .tag import Action, Player, TagState
except ImportError:
    from reinforcement_learning import BaseAgent, EpisodeConfig
    from tag import Action, Player, TagState


GROUP_NAME = "bello"


class Agent(BaseAgent):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        # Initialize your model and hyperparameters here. Use self.rng for
        # reproducible randomness. The supplied framework runs training.

        # q table (state, action) -> value
        self.q: dict[tuple[int, int], float] = {}

        # q learning settings
        self.alpha = 0.8
        self.gamma = 0.9
        self.epsilon = 0.1

        self.episodes = 0

    def init_episode(self) -> EpisodeConfig:
        """Return EpisodeConfig(side_length=..., n_opponents=..., max_steps=...).

        All three fields are positive integers. n_opponents + 1 must fit in
        side_length ** 2 cells. max_steps counts total player turns.
        Reset episode history here and preserve your learned model.
        """
        self.episodes += 1

        return EpisodeConfig(
            side_length = 4,
            n_opponents = 3,
            max_steps = 100
        )

    def encode_state(self, state: TagState, player: Player) -> int:
        """Convert the board from player's perspective to a stable Python int.

        Include the features your policy and model need. Keep this conversion
        free of side effects and retain encoded integers in your model and history.
        """
        value = 0
        base = state.n_rows * state.n_cols + 1

        # stores every players position
        for position in state.positions:
            value = value * base + (
                position[0] * state.n_cols + position[1] + 1
            )

        # store who is IT
        value = value * (len(state.players) + 1)
        value += state.players.index(state.tagged_player) + 1

        # Store whose turn it is.
        value = value * (len(state.players) + 1)
        value += state.players.index(state.next_player) + 1

        # Store whether someone was just tagged.
        value = value * 2 + int(state.just_tagged)

        # Store the turn number.
        value = value * (state.max_turns + 1) + state.turn

        # Store which player we are controlling.
        value = value * (len(state.players) + 1)
        value += state.players.index(player) + 1

        return int(value)

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
        old_score = state.score_of(player)
        new_score = next_state.score_of(player)

        # +10 for tagging someone.
        reward = 10.0 * (
            new_score.tags - old_score.tags
        )

        # -1 for spending a turn as It.
        reward -= 1.0 * (
            new_score.it_turns - old_score.it_turns
        )

        return float(reward)

    def select_action(
        self, state: int, legal_actions: tuple[Action, ...], training: bool,
    ) -> Action | None:
        """Choose from legal_actions using the encoded integer state.

        Use exploration when training=True and your evaluation policy when
        training=False. Return None when legal_actions is empty.
        """
        # No actions means the game is over.
        if not legal_actions:
            return None

        # Explore during training.
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # Otherwise choose the action with the highest Q-value.
        best_value = max(
            self.q.get((state, int(action)), 0.0)
            for action in legal_actions
        )

        best_actions = [
            action
            for action in legal_actions
            if self.q.get((state, int(action)), 0.0) == best_value
        ]

        return self.rng.choice(best_actions)

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
        key = (state, int(action))

        old_value = self.q.get(key, 0.0)

        # If the game is over, there is no future value.
        if terminal or not next_actions:
            target = reward
        else:
            next_value = max(
                self.q.get((next_state, int(a)), 0.0)
                for a in next_actions
            )

            target = reward + self.gamma * next_value

        # Q-learning formula.
        self.q[key] = old_value + self.alpha * (
            target - old_value
        )