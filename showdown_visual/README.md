# Local Showdown Visualizer

This folder is a standalone adapter that connects your trained policy to a local Pokemon Showdown server and prints a spectator URL so you can watch the battle in the browser.

## Quick start

```bash
python showdown_visual/run_showdown_battle.py \
  --model outputs/models/maskable_ppo_warmstart.zip \
  --battle-format gen9doublesou \
  --team teams/gen9dou_fixed.txt \
  --server-url http://localhost:8000
```

You will see a line like:

```
[spectate] http://localhost:8000/battle-gen9doublesou-123456
```

Open that URL in your browser to watch the match.

## Common variants

- Behavior cloning checkpoint:
  ```bash
  python showdown_visual/run_showdown_battle.py \
    --model outputs/models/bc_policy.pt \
    --stats outputs/models/bc_stats.json
  ```

- Run multiple battles:
  ```bash
  python showdown_visual/run_showdown_battle.py --model ... --battles 3
  ```

- Show agent view in terminal:
  ```bash
  python showdown_visual/run_showdown_battle.py --model ... --log-level summary
  ```

- Mirror match (policy vs itself):
  ```bash
  python showdown_visual/run_showdown_battle.py --model ... --opponent mirror
  ```

- Policy vs policy:
  ```bash
  python showdown_visual/run_showdown_battle.py \
    --model outputs/models/maskable_ppo_warmstart.zip \
    --opponent policy \
    --opponent-model outputs/models/maskable_ppo_scratch.zip
  ```

## Notes

- Default format is `gen9doublesou`.
- The server must already be running at `http://localhost:8000`.
- If you use a different team for the opponent, pass `--opponent-team`.
