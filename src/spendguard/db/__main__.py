from spendguard import settings
from spendguard.branch_names import database_name_for_branch
from spendguard.db.database import ensure_branch_database
from spendguard.structured_logging import configure_logging, log_event


def main() -> None:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    ensure_branch_database(loaded_settings)
    log_event('database_ready', database=database_name_for_branch(loaded_settings.branch_name))


if __name__ == '__main__':
    main()
