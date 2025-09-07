import typer

app = typer.Typer()


@app.command()
def main(n_games: int = 1000):
    print("Offline evaluation stub with", n_games, "games.")


if __name__ == "__main__":
    app()
