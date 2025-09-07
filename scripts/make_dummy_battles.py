from pathlib import Path

RUN_DIR = Path("runs/latest/battles")
RUN_DIR.mkdir(parents=True, exist_ok=True)

samples = {
    "battle-001.log": [
        "|player|p1|RLBot|1000",
        "|player|p2|MaxDamage|1000",
        "|turn|1",
        "|move|p1a: MonA|Protect|p1a: MonA",
        "|move|p1b: MonB|Fake Out|p2a: OppA",
        "|move|p2a: OppA|Make It Rain|p1a: MonA",
        "|win|RLBot",
    ],
    "battle-002.log": [
        "|player|p1|RLBot|1000",
        "|player|p2|Random|1000",
        "|turn|1",
        "|switch|p1a: MonC|MonC, L80|100/100",
        "|move|p2a: OppC|Thunderbolt|p1a: MonC",
        "|turn|2",
        "|move|p1a: MonC|Heat Wave|p2a: OppC",
        "|win|Random",
    ],
}

for name, lines in samples.items():
    (RUN_DIR / name).write_text("\n".join(lines))
print(f"Wrote {len(samples)} dummy battle logs to {RUN_DIR}")
