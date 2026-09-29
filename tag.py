"""Immutable multiplayer tag environment for CS 4341 Assignment 3."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence


Player = str
Position = tuple[int, int]


class Action(IntEnum):
    """Every action available in the tag environment."""

    WAIT = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3
    MOVE_EAST = 4
    TAG_NORTH = 5
    TAG_SOUTH = 6
    TAG_WEST = 7
    TAG_EAST = 8


WAIT = Action.WAIT
MOVE_NORTH = Action.MOVE_NORTH
MOVE_SOUTH = Action.MOVE_SOUTH
MOVE_WEST = Action.MOVE_WEST
MOVE_EAST = Action.MOVE_EAST
TAG_NORTH = Action.TAG_NORTH
TAG_SOUTH = Action.TAG_SOUTH
TAG_WEST = Action.TAG_WEST
TAG_EAST = Action.TAG_EAST

MOVE_ACTIONS = (
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_WEST,
    MOVE_EAST,
)
TAG_ACTIONS = (
    TAG_NORTH,
    TAG_SOUTH,
    TAG_WEST,
    TAG_EAST,
)

_DIRECTION = {
    MOVE_NORTH: (-1, 0),
    MOVE_SOUTH: (1, 0),
    MOVE_WEST: (0, -1),
    MOVE_EAST: (0, 1),
    TAG_NORTH: (-1, 0),
    TAG_SOUTH: (1, 0),
    TAG_WEST: (0, -1),
    TAG_EAST: (0, 1),
}
_MOVE_TO_TAG = {
    MOVE_NORTH: TAG_NORTH,
    MOVE_SOUTH: TAG_SOUTH,
    MOVE_WEST: TAG_WEST,
    MOVE_EAST: TAG_EAST,
}


@dataclass(frozen=True, slots=True)
class PlayerScore:
    """The two tournament statistics accumulated by one player."""

    it_turns: int = 0
    tags: int = 0

    def __post_init__(self) -> None:
        if type(self.it_turns) is not int or self.it_turns < 0:
            raise ValueError("it_turns must be a nonnegative integer.")
        if type(self.tags) is not int or self.tags < 0:
            raise ValueError("tags must be a nonnegative integer.")


@dataclass(frozen=True, slots=True)
class TagState:
    """A complete immutable tag-board state.

    ``positions`` and ``scores`` use the same ordering as ``players``.
    Convenience methods provide player-based lookup without changing the
    state's immutable, hashable representation.
    """

    n_rows: int
    n_cols: int
    players: tuple[Player, ...]
    positions: tuple[Position, ...]
    scores: tuple[PlayerScore, ...]
    next_player_index: int
    tagged_player_index: int
    just_tagged: bool
    turn: int
    max_turns: int

    def __post_init__(self) -> None:
        if type(self.n_rows) is not int or self.n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")
        if type(self.n_cols) is not int or self.n_cols <= 0:
            raise ValueError("n_cols must be a positive integer.")
        if len(self.players) < 2 or len(set(self.players)) != len(self.players):
            raise ValueError("players must contain at least two unique names.")
        if any(not isinstance(player, str) or not player for player in self.players):
            raise ValueError("Every player name must be a nonempty string.")
        if len(self.positions) != len(self.players):
            raise ValueError("positions must contain one position per player.")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("Player positions must be distinct.")
        if any(not self._on_board(position) for position in self.positions):
            raise ValueError("Every player position must be on the board.")
        if len(self.scores) != len(self.players):
            raise ValueError("scores must contain one score per player.")
        if not 0 <= self.next_player_index < len(self.players):
            raise ValueError("next_player_index is out of range.")
        if not 0 <= self.tagged_player_index < len(self.players):
            raise ValueError("tagged_player_index is out of range.")
        if type(self.just_tagged) is not bool:
            raise ValueError("just_tagged must be a bool.")
        if type(self.turn) is not int or self.turn < 0:
            raise ValueError("turn must be a nonnegative integer.")
        if type(self.max_turns) is not int or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")
        if self.turn > self.max_turns:
            raise ValueError("turn cannot exceed max_turns.")

    def _on_board(self, position: Position) -> bool:
        return (
            isinstance(position, tuple)
            and len(position) == 2
            and type(position[0]) is int
            and type(position[1]) is int
            and 0 <= position[0] < self.n_rows
            and 0 <= position[1] < self.n_cols
        )

    @property
    def next_player(self) -> Player:
        """Return the player whose turn it is."""
        return self.players[self.next_player_index]

    @property
    def tagged_player(self) -> Player:
        """Return the player who is currently It."""
        return self.players[self.tagged_player_index]

    @property
    def player_positions(self) -> dict[Player, Position]:
        """Return a new player-to-position dictionary."""
        return dict(zip(self.players, self.positions))

    @property
    def player_scores(self) -> dict[Player, PlayerScore]:
        """Return a new player-to-score dictionary."""
        return dict(zip(self.players, self.scores))

    def position_of(self, player: Player) -> Position:
        """Return ``player``'s position."""
        return self.positions[self.players.index(player)]

    def score_of(self, player: Player) -> PlayerScore:
        """Return ``player``'s score."""
        return self.scores[self.players.index(player)]


class Tag:
    """Deterministic rules and transitions for a finite tag game."""

    def __init__(self, initial: TagState) -> None:
        self.initial = initial

    @classmethod
    def random(
        cls,
        *,
        n_rows: int,
        n_cols: int,
        players: Sequence[Player],
        max_turns: int,
        seed: int,
        tagged_player: Player | None = None,
    ) -> Tag:
        """Create a reproducible game with distinct random starting positions."""
        player_tuple = tuple(players)
        if len(player_tuple) < 2:
            raise ValueError("A tag game requires at least two players.")
        if n_rows * n_cols < len(player_tuple):
            raise ValueError("The board needs at least one cell per player.")
        if len(set(player_tuple)) != len(player_tuple):
            raise ValueError("Player names must be unique.")

        rng = random.Random(seed)
        cells = [
            (row, column)
            for row in range(n_rows)
            for column in range(n_cols)
        ]
        positions = tuple(rng.sample(cells, len(player_tuple)))
        if tagged_player is None:
            tagged_player_index = rng.randrange(len(player_tuple))
        else:
            try:
                tagged_player_index = player_tuple.index(tagged_player)
            except ValueError as error:
                raise ValueError("tagged_player must be in players.") from error

        initial = TagState(
            n_rows=n_rows,
            n_cols=n_cols,
            players=player_tuple,
            positions=positions,
            scores=tuple(PlayerScore() for _ in player_tuple),
            next_player_index=0,
            tagged_player_index=tagged_player_index,
            just_tagged=True,
            turn=0,
            max_turns=max_turns,
        )
        return cls(initial)

    def actions(self, state: TagState) -> tuple[Action, ...]:
        """Return legal actions in deterministic compass order."""
        self._validate_state(state)
        if self.is_terminal(state):
            return ()

        position = state.positions[state.next_player_index]
        occupied = {
            other_position: index
            for index, other_position in enumerate(state.positions)
            if index != state.next_player_index
        }
        is_it = state.next_player_index == state.tagged_player_index
        actions: list[Action] = []

        for move_action in MOVE_ACTIONS:
            destination = _offset(position, _DIRECTION[move_action])
            if not _on_board(state, destination):
                continue
            if destination not in occupied:
                actions.append(move_action)
            elif is_it and not state.just_tagged:
                actions.append(_MOVE_TO_TAG[move_action])

        if is_it or not actions:
            actions.append(WAIT)
        return tuple(actions)

    def result(self, state: TagState, action: Action) -> TagState:
        """Return the state produced by one legal action."""
        self._validate_state(state)
        if isinstance(action, bool):
            raise ValueError(f"Unknown tag action: {action!r}.")
        try:
            normalized_action = Action(action)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unknown tag action: {action!r}.") from error
        if normalized_action not in self.actions(state):
            raise ValueError(
                f"Action {normalized_action.name} is not legal in this state."
            )

        current = state.next_player_index
        positions = list(state.positions)
        scores = list(state.scores)
        tagged_player_index = state.tagged_player_index
        just_tagged = state.just_tagged

        if normalized_action == WAIT:
            if current == tagged_player_index:
                scores[current] = PlayerScore(
                    it_turns=scores[current].it_turns + 1,
                    tags=scores[current].tags,
                )
                just_tagged = False
        elif normalized_action in MOVE_ACTIONS:
            positions[current] = _offset(
                positions[current],
                _DIRECTION[normalized_action],
            )
            if current == tagged_player_index:
                scores[current] = PlayerScore(
                    it_turns=scores[current].it_turns + 1,
                    tags=scores[current].tags,
                )
                just_tagged = False
        else:
            destination = _offset(
                positions[current],
                _DIRECTION[normalized_action],
            )
            tagged_player_index = positions.index(destination)
            scores[current] = PlayerScore(
                it_turns=scores[current].it_turns,
                tags=scores[current].tags + 1,
            )
            just_tagged = True

        return TagState(
            n_rows=state.n_rows,
            n_cols=state.n_cols,
            players=state.players,
            positions=tuple(positions),
            scores=tuple(scores),
            next_player_index=(current + 1) % len(state.players),
            tagged_player_index=tagged_player_index,
            just_tagged=just_tagged,
            turn=state.turn + 1,
            max_turns=state.max_turns,
        )

    def is_terminal(self, state: TagState) -> bool:
        """Return whether the fixed turn budget has been exhausted."""
        self._validate_state(state)
        return state.turn >= state.max_turns

    def _validate_state(self, state: TagState) -> None:
        if not isinstance(state, TagState):
            raise TypeError("Tag methods require a TagState.")


def _offset(position: Position, direction: Position) -> Position:
    return position[0] + direction[0], position[1] + direction[1]


def _on_board(state: TagState, position: Position) -> bool:
    return (
        0 <= position[0] < state.n_rows
        and 0 <= position[1] < state.n_cols
    )


def format_state(state: TagState) -> str:
    """Return a compact board and score display."""
    width = max(3, max(len(player) for player in state.players) + 2)
    grid = [["." for _ in range(state.n_cols)] for _ in range(state.n_rows)]
    for index, (player, position) in enumerate(
        zip(state.players, state.positions)
    ):
        label = f"[{player}]" if index == state.tagged_player_index else player
        grid[position[0]][position[1]] = label
    lines = [
        f"turn {state.turn}/{state.max_turns}; "
        f"next={state.next_player}; it={state.tagged_player}",
        *(
            " ".join(f"{cell:^{width}}" for cell in row)
            for row in grid
        ),
        "scores: "
        + ", ".join(
            f"{player}(it={score.it_turns}, tags={score.tags})"
            for player, score in zip(state.players, state.scores)
        ),
    ]
    return "\n".join(lines)


def state_to_dict(state: TagState) -> dict[str, object]:
    """Serialize a state for the isolated grader and tournament workers."""
    return {
        "n_rows": state.n_rows,
        "n_cols": state.n_cols,
        "players": list(state.players),
        "positions": [list(position) for position in state.positions],
        "scores": [
            {"it_turns": score.it_turns, "tags": score.tags}
            for score in state.scores
        ],
        "next_player_index": state.next_player_index,
        "tagged_player_index": state.tagged_player_index,
        "just_tagged": state.just_tagged,
        "turn": state.turn,
        "max_turns": state.max_turns,
    }


def state_from_dict(data: object) -> TagState:
    """Deserialize and validate a state."""
    if not isinstance(data, Mapping):
        raise TypeError("Serialized state must be a mapping.")
    players = _string_tuple(data.get("players"), "players")
    raw_positions = data.get("positions")
    if not isinstance(raw_positions, list):
        raise TypeError("positions must be a list.")
    positions: list[Position] = []
    for value in raw_positions:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(coordinate) is not int for coordinate in value)
        ):
            raise TypeError("Every position must be a two-integer list.")
        positions.append((value[0], value[1]))

    raw_scores = data.get("scores")
    if not isinstance(raw_scores, list):
        raise TypeError("scores must be a list.")
    scores: list[PlayerScore] = []
    for value in raw_scores:
        if not isinstance(value, Mapping):
            raise TypeError("Every score must be a mapping.")
        scores.append(
            PlayerScore(
                it_turns=_exact_int(value.get("it_turns"), "it_turns"),
                tags=_exact_int(value.get("tags"), "tags"),
            )
        )

    return TagState(
        n_rows=_exact_int(data.get("n_rows"), "n_rows"),
        n_cols=_exact_int(data.get("n_cols"), "n_cols"),
        players=players,
        positions=tuple(positions),
        scores=tuple(scores),
        next_player_index=_exact_int(
            data.get("next_player_index"),
            "next_player_index",
        ),
        tagged_player_index=_exact_int(
            data.get("tagged_player_index"),
            "tagged_player_index",
        ),
        just_tagged=_exact_bool(data.get("just_tagged"), "just_tagged"),
        turn=_exact_int(data.get("turn"), "turn"),
        max_turns=_exact_int(data.get("max_turns"), "max_turns"),
    )


def action_to_data(action: Action | None) -> int | None:
    """Serialize an action as its stable integer code."""
    if action is None:
        return None
    if isinstance(action, bool):
        raise ValueError(f"Invalid tag action: {action!r}.")
    try:
        return int(Action(action))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid tag action: {action!r}.") from error


def action_from_data(data: object) -> Action | None:
    """Deserialize an action from its stable integer code."""
    if data is None:
        return None
    if type(data) is not int:
        raise TypeError("Serialized action must be an integer or null.")
    try:
        return Action(data)
    except ValueError as error:
        raise ValueError(f"Unknown serialized action code: {data}.") from error


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise TypeError(f"{name} must be a list of strings.")
    return tuple(value)


__all__ = [
    "Action",
    "MOVE_ACTIONS",
    "MOVE_EAST",
    "MOVE_NORTH",
    "MOVE_SOUTH",
    "MOVE_WEST",
    "Player",
    "PlayerScore",
    "Position",
    "TAG_ACTIONS",
    "TAG_EAST",
    "TAG_NORTH",
    "TAG_SOUTH",
    "TAG_WEST",
    "Tag",
    "TagState",
    "WAIT",
    "action_from_data",
    "action_to_data",
    "format_state",
    "state_from_dict",
    "state_to_dict",
]
