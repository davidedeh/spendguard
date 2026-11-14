from spendguard import settings
from spendguard.db.database import branch_database_name, ensure_branch_database
from spendguard.structured_logging import configure_logging, log_event


def main() -> None:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    ensure_branch_database(loaded_settings)
    log_event('database_ready', database=branch_database_name(loaded_settings))


if __name__ == '__main__':
    main()
