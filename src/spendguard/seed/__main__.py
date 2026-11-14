import sys
from pathlib import Path

import psycopg

from spendguard import settings
from spendguard.branch_names import database_name_for_branch
from spendguard.db.database import ensure_branch_database
from spendguard.seed.loader import SEED_DIRECTORY, load_seed_data, read_seed_directory
from spendguard.structured_logging import configure_logging, log_event


def main(arguments: list[str]) -> None:
    directory = Path(arguments[0]) if arguments else SEED_DIRECTORY
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    seed_data = read_seed_directory(directory)
    with psycopg.connect(ensure_branch_database(loaded_settings)) as connection:
        load_seed_data(connection, seed_data)
    log_event('seed_loaded', database=database_name_for_branch(loaded_settings.branch_name))


if __name__ == '__main__':
    main(sys.argv[1:])
