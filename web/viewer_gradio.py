from os import getenv
from pathlib import Path

import gradio as gr

REPLAY_DIR = Path(getenv("REPLAY_DIR", "replays"))
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def list_battles():
    return sorted([f.name for f in REPLAY_DIR.glob("*.log")])


def read_battle(file_name: str, tail: int = 200):
    if not file_name:
        return ""
    p = REPLAY_DIR / file_name
    if not p.exists():
        return "No such battle file"
    lines = p.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-tail:])


with gr.Blocks(title="Showdown RL viewer") as demo:
    gr.Markdown("# Battle Replays (tail view)")
    with gr.Row():
        refresh = gr.Button("Refresh list")
        battle = gr.Dropdown(choices=list_battles(), label="Battle file")
    out = gr.Textbox(label="Tail", lines=24)
    tail = gr.Slider(100, 1000, value=200, step=50, label="Lines to show")

    def refresh_choices():
        return gr.Dropdown(choices=list_battles())

    refresh.click(fn=refresh_choices, outputs=battle)
    battle.change(fn=read_battle, inputs=[battle, tail], outputs=out)
    tail.change(fn=read_battle, inputs=[battle, tail], outputs=out)

if __name__ == "__main__":
    demo.launch()
