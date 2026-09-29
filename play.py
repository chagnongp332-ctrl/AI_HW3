"""Train your agent, then play a local evaluation game: python play.py."""
from __future__ import annotations

import argparse
import time

from main import Agent
from reinforcement_learning import EpisodeConfig, choose_action, train_agent
from tag import Tag, format_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--updates", type=int, default=1_000,
        help="total self-play model updates (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=4341)
    parser.add_argument(
        "--side-length", type=int, default=6,
        help="evaluation board side length (default: 6)",
    )
    parser.add_argument(
        "--opponents", type=int, default=3,
        help="other evaluation players (default: 3)",
    )
    args = parser.parse_args()
    if not 0 <= args.updates <= 1_000_000:
        parser.error("--updates must be between 0 and 1000000")
    try:
        config = EpisodeConfig(
            side_length=args.side_length,
            n_opponents=args.opponents,
            max_steps=100 * (args.opponents + 1),
        )
    except ValueError as error:
        parser.error(str(error))
    started = time.perf_counter()
    agent = train_agent(Agent(seed=args.seed), seed=args.seed, updates=args.updates)
    elapsed = time.perf_counter() - started
    print(f"Training: {args.updates:,} updates in {elapsed:.2f} seconds")
    problem = Tag.random(
        n_rows=config.side_length,
        n_cols=config.side_length,
        players=tuple(f"P{i}" for i in range(config.n_opponents + 1)),
        max_turns=config.max_steps,
        seed=args.seed + 1,
    )
    state = problem.initial
    while not problem.is_terminal(state):
        started = time.perf_counter()
        action = choose_action(agent, state, training=False)
        if time.perf_counter() - started >= 1.0:
            raise RuntimeError("Encoding and action selection exceeded the one-second limit")
        state = problem.result(state, action)
    print(format_state(state))
    for player in state.players:
        score = state.score_of(player)
        print(f"{player}: {score.it_turns} turns ending as It; {score.tags} successful tags")


if __name__ == "__main__":
    main()
