import sys

from spendguard import settings
from spendguard.branch_names import database_name_for_branch
from spendguard.db.database import (
    NotALocalDatabaseError,
    drop_branch_database,
    ensure_branch_database,
)
from spendguard.structured_logging import configure_logging, log_event

DROP_FIRST_FLAG = '--drop-first'


def main(arguments: list[str]) -> None:
    if arguments not in ([], [DROP_FIRST_FLAG]):
        sys.exit(f'usage: python -m spendguard.db [{DROP_FIRST_FLAG}]')
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    database_name = database_name_for_branch(loaded_settings.branch_name)
    if arguments == [DROP_FIRST_FLAG]:
        try:
            drop_branch_database(loaded_settings)
        except NotALocalDatabaseError as error:
            sys.exit(str(error))
        log_event('database_dropped', database=database_name)
    ensure_branch_database(loaded_settings)
    log_event('database_ready', database=database_name)


if __name__ == '__main__':
    main(sys.argv[1:])
