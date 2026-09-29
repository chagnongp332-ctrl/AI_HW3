# CS 4341 Assignment 3: learn to play tag

Start with `cs4341_assignment_3.pdf` for the game rules, agent interface, training budget, submission requirements, and grading rubric. Use this guide to set up and run your code.

## Set up

Use Python 3.10 or newer. Extract this entire bundle into one directory. From that directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS/Linux: `source .venv/bin/activate`  
Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install the public test dependency:

```bash
python -m pip install -r requirements.txt
```

Use `python3` in place of `python` if that is your Python 3 command. Your submitted implementation uses Python's standard library; pytest runs the supplied tests.

## Implement

Edit `main.py`: set `GROUP_NAME`, initialize your model in `Agent.__init__(seed)`, and complete `init_episode`, `encode_state`, `calculate_reward`, `select_action`, and `update_model`. The method docstrings summarize each task; the handout defines the full contract. The starter raises `NotImplementedError` until you complete these methods.

`encode_state` converts a board into one Python `int`. Your policy and model updates use encoded integers, with legal actions and terminal flags supplied by the framework. Your reward function receives the complete board states before and after a player's decision interval. The handout explains these inputs and their timing.

## Test and run

```bash
python -m pytest -q test_cases.py
python play.py --updates 1000
```

The 20 public tests check compatibility with the supplied framework. The simulation trains your agent through self-play, then uses `training=False` for every player in one evaluation game and prints its final board and scores.

Run the full training budget with:

```bash
python play.py --updates 1000000
```

The local evaluation defaults to a 6-by-6 board, three opponents, and 100 rounds. Each round gives every player one turn. Set `--side-length` and `--opponents` to try other evaluation configurations, and `--seed` to make a run reproducible:

```bash
python play.py --updates 1000 --side-length 8 --opponents 5 --seed 42
```

These board options affect evaluation. Choose your training boards and episode lengths in `init_episode()`.

## Files

- `main.py`: your agent implementation.
- `reinforcement_learning.py`: abstract interface, episode configuration, and self-play training loop.
- `tag.py`: game states, legal actions, and transitions.
- `test_cases.py`: 20 public tests.
- `play.py`: local training and evaluation command.
- `requirements.txt`: test dependency.
- `cs4341_assignment_3.pdf`: assignment handout.
