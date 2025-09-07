import typer

app = typer.Typer()


@app.command()
def main(budget_games: int = 200):
    print("Ladder evaluation stub with", budget_games, "ranked games.")


if __name__ == "__main__":
    app()
