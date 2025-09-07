# RL agent for Pokémon Showdown Gen 9 Doubles

Start: Thu 11 September 2025  
Hand in: Thu 18 December 2025 at 10:00 CET

## What we are building
PPO-based recurrent agent in TensorFlow/Keras playing **Gen 9 Doubles (OU)** via **poke-env**. Train mostly on a local Showdown simulator. Fine-tune on a labeled bot account with polite rate limits. Evaluate offline vs heuristics and on ladder rating.

PoC: https://github.com/victorwp288/poke-rl-demo

## Quick start
```bash
pip install -r requirements.txt

# test setup
python tests/smoke_test_env.py
```

## Web viewer
This repo includes a very light Gradio UI that tails battle log files under `runs/latest/battles/`.


Tip: if you don't have logs yet, generate a couple of dummy ones to test the UI:
```bash
python scripts/make_dummy_battles.py
```

## Optional SQLite database
If you decide to store run metadata, and create a local DB:
```bash
python -c "from db.schema import create_db; create_db('runs.sqlite')"
```

## Repo layout
```
poke_rl_doubles/
  src/
    envs/
      __init__.py
      wrappers.py
    models/
      __init__.py
      policy.py
    algo/
      __init__.py
      ppo_tf.py
    agents/
      __init__.py
      rl_player.py
      heuristic_players.py
    io/
      __init__.py
      replay_parser.py
      logging_utils.py
  scripts/
    train_selfplay.py
    train_imitation.py
    eval_offline.py
    eval_ladder.py
    make_dummy_battles.py
  web/
    viewer_gradio.py
  db/
    schema.py
  teams/
    gen9dou_fixed.txt
  configs/
    default.yml
  tests/
    test_masks.py
    smoke_test_env.py
  docs/
    reading.md
  requirements.txt
  requirements_web.txt
  requirements_db.txt
  README.md
```
