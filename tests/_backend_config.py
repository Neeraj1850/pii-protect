"""
Shared local-backend connection settings for Postgres/Redis-backed tests.

Not a conftest — a plain module imported by the conftests that need it
(tests/unit/storage/conftest.py, tests/integration/conftest.py) so the
DSN/URL resolution logic lives in exactly one place.

Override via TEST_POSTGRES_DSN / TEST_REDIS_URL to point at a different
instance (e.g. in CI). Defaults assume a local Postgres/Redis reachable
with the current OS user and no password, matching the dev environment
this suite ships with.
"""
import getpass
import os

DEFAULT_POSTGRES_DSN = f"postgresql://{getpass.getuser()}@localhost:5432/postgres"
POSTGRES_DSN = os.environ.get("TEST_POSTGRES_DSN", DEFAULT_POSTGRES_DSN)

# DB 15 keeps test data out of the default DB 0 an application would use.
DEFAULT_REDIS_URL = "redis://localhost:6379/15"
REDIS_URL = os.environ.get("TEST_REDIS_URL", DEFAULT_REDIS_URL)
