import os

import pytest

from utils import db


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    os.environ["DB_FILE"] = "test_logs.db"
    db.DB_FILE = "test_logs.db"
    db.init_db()
    yield
    if os.path.exists("test_logs.db"):
        os.remove("test_logs.db")
    print("Test database setup complete.")
