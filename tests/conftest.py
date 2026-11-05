import pandas as pd
import pytest

from cost_analytics_agent.db import QueryRunner, build_database
from cost_analytics_agent.generate import generate
from cost_analytics_agent.tools import Toolbox


@pytest.fixture(scope="session")
def dataset():
    return generate()


@pytest.fixture(scope="session")
def focus(dataset):
    return dataset[0]


@pytest.fixture(scope="session")
def incidents(dataset):
    return {inc["id"]: inc for inc in dataset[1]}


@pytest.fixture(scope="session")
def runner(focus, tmp_path_factory):
    root = tmp_path_factory.mktemp("db")
    parquet = root / "focus_cost.parquet"
    focus.to_parquet(parquet, index=False)
    build_database(parquet, root / "cost.duckdb")
    runner = QueryRunner(root / "cost.duckdb")
    yield runner
    runner.close()


@pytest.fixture(scope="session")
def toolbox(runner):
    return Toolbox(runner)


@pytest.fixture(scope="session")
def daily_cost(runner):
    return runner.con.execute("select * from daily_cost").df()


@pytest.fixture(scope="session")
def daily_total(toolbox) -> pd.Series:
    return toolbox.daily_series()
