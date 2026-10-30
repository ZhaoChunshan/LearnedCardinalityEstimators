# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Added for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Addition: shared terminal and file logging for the local GenJoin scripts.

"""
Unified logging config. Use in any script:

    from src.utils.logging_config import setup_logging
    logger = setup_logging('gen_data', log_dir='logs')

Logs to terminal + file: logs/<script>_YYYYMMDD-HHMMSS.log
"""

import os
import sys
import time
import logging


def setup_logging(script_name: str, log_dir: str = 'logs') -> logging.Logger:
    """Init logging to both terminal and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"{script_name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger
