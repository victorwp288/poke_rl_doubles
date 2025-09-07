import typer

app = typer.Typer()


@app.command()
def main(dataset_path: str = "data/replays.jsonl"):
    print("Imitation training stub. Load", dataset_path, "then train a policy.")


if __name__ == "__main__":
    app()
