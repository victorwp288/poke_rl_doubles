from datetime import datetime

from sqlmodel import Field, SQLModel, create_engine


class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str | None = None


def create_db(path: str = "runs.sqlite"):
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine)
    print(f"Created {path}")
    return engine
