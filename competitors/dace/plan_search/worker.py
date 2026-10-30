"""Worker processes for the exhaustive plan search.

Each worker process owns its own DB connection (a PilotDataInteractor backed by
a sqlalchemy connection pool) and repeatedly: pushes a Leading-only hint,
EXPLAINs the query, and reports the physical plan back if PostgreSQL honored
the join order. The connection is created once at worker start and destroyed
promptly when the worker exits.

Deliberately no torch / DACE import here: the cost model lives in the parent
process only.
"""

import logging
import time
from typing import Optional

from plan_generator.query_hint import Hint

logger = logging.getLogger(__name__)


def _make_interactor(db: str):
    """Create a DB interactor and register the physical-plan pull anchor once.

    The interactor's connection is reused across all tasks of this worker
    (execute(..., is_reset=False) so pilotscope does not re-open a connection
    on every EXPLAIN).
    """
    from pilotscope.PilotConfig import PostgreSQLConfig
    from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
    from plan_generator.connection_args import DB_CONNECTION_ARGS

    cfg = DB_CONNECTION_ARGS
    config = PostgreSQLConfig(
        db_host=cfg["host"],
        db_port=cfg["port"],
        db_user=cfg["user"],
        db_user_pwd=cfg["password"],
        db=db,
    )
    interactor = PilotDataInteractor(config)
    interactor.pull_physical_plan()
    return interactor


def _close_interactor(interactor) -> None:
    """Promptly destroy the worker's DB connection(s)."""
    try:
        controller = interactor.db_controller
        controller._disconnect()  # close the thread-local sqlalchemy connection
        controller.engine.dispose()  # close every pooled connection
    except Exception:
        logger.warning("Failed to close DB connection cleanly", exc_info=True)


def _normalize_leading(leading: str) -> str:
    return "".join(leading.split())


def evaluate_one(interactor, sql: str, leading: str) -> Optional[dict]:
    """Push a Leading-only hint, EXPLAIN the query, return the physical plan.

    Returns None if PostgreSQL rejected the hint (join order not honored) or
    the EXPLAIN failed.
    """
    hint_str = Hint(leading, set()).to_pg_hint()
    interactor.push_pg_hint_comment(hint_str)
    data = interactor.execute(sql, is_reset=False)
    if data is None or data.physical_plan is None:
        return None

    plan = data.physical_plan
    actual_leading = Hint.from_pg_physical_plan_json(plan).leading_hint
    if _normalize_leading(actual_leading) != _normalize_leading(leading):
        return None
    return plan


def worker_main(task_queue, result_queue, db: str) -> None:
    """Entry point of each worker process. Consumes tasks until a None sentinel."""
    logger.info("Worker for db=%s started (pid=%d)", db, __import__("os").getpid())
    interactor = _make_interactor(db)
    try:
        while True:
            task = task_queue.get()
            if task is None:  # shutdown sentinel
                break
            task_id, sql, leading = task
            start = time.perf_counter()
            try:
                plan = evaluate_one(interactor, sql, leading)
            except Exception as e:
                logger.warning("Worker task %s failed: %s", task_id, e)
                plan = None
            elapsed = time.perf_counter() - start
            if plan is None:
                logger.debug("Task %s rejected/skipped in %.3fs", task_id, elapsed)
            result_queue.put((task_id, plan))
    finally:
        _close_interactor(interactor)
        logger.info("Worker for db=%s exiting (pid=%d)", db, __import__("os").getpid())
