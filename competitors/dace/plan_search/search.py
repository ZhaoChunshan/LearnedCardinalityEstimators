"""Exhaustive join-order search using the C++ enumerator + DACE cost model.

Pipeline for one query:
  1. Parse the query into a join graph and enumerate join orders (C++).
  2. For each join order, ask PostgreSQL for the physical plan. This round-trip
     is the bottleneck, so it runs in a pool of worker processes: each worker
     owns its own DB connection, and tasks/results flow over two
     multiprocessing.Queue objects.
  3. In the parent process, score the accepted physical plans with the DACE
     cost model and pick the cheapest one.

The worker pool persists across queries (connections are per-process and
reused), so process/connection startup cost is amortized. The DACE model is
loaded lazily in the parent only.
"""

import logging
import multiprocessing as mp
import os
import queue
import time
from typing import Dict, Iterator, List, Optional, Tuple

from plan_generator.query_hint import Hint
from plan_generator.sql_query import Query

from plan_search import enumerator
from plan_search.worker import worker_main

logger = logging.getLogger(__name__)


class WorkerPoolError(RuntimeError):
    """Raised when a worker process dies mid-batch (results would be missing)."""


def _default_num_workers() -> int:
    cpus = os.cpu_count() or 4
    return max(2, min(64, cpus))


class WorkerPool:
    """A pool of DB-connected worker processes fed by two Queues.

    Task format on the input queue:  (task_id:int, sql:str, leading:str)
    Result format on the output queue: (task_id:int, physical_plan:dict|None)
    A None sentinel shuts a worker down.
    """

    def __init__(self, db: str, num_workers: int, result_timeout: float = 120.0):
        self.db = db
        self.num_workers = num_workers
        self.result_timeout = result_timeout
        self._closed = False

        # spawn context: child processes start fresh, so a CUDA-context already
        # initialized in the parent (the DACE model) never leaks into workers.
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.processes = [
            ctx.Process(target=worker_main, args=(self.task_queue, self.result_queue, db), daemon=True)
            for _ in range(num_workers)
        ]
        for p in self.processes:
            p.start()
        logger.info("Started %d workers for db=%s", num_workers, db)

    def iter_results(self, tasks: List[Tuple[int, str, str]]) -> Iterator[Tuple[int, Optional[dict]]]:
        """Put all tasks, then yield (task_id, physical_plan) as results arrive.

        If a worker dies before producing all results, raises WorkerPoolError
        (the pool is closed; the caller should create a fresh one).
        """
        if self._closed:
            raise WorkerPoolError("worker pool is closed")
        for t in tasks:
            self.task_queue.put(t)

        n = len(tasks)
        seen: set = set()
        while len(seen) < n:
            try:
                task_id, plan = self.result_queue.get(timeout=self.result_timeout)
            except queue.Empty:
                alive = sum(p.is_alive() for p in self.processes)
                if alive < self.num_workers:
                    self.close()
                    raise WorkerPoolError(
                        f"{self.num_workers - alive} worker(s) died; pool closed"
                    )
                logger.warning(
                    "No result for %ss although all workers alive; retrying", self.result_timeout
                )
                continue
            if task_id in seen:
                continue  # duplicate result from a retried task; ignore
            seen.add(task_id)
            yield task_id, plan

    def close(self) -> None:
        """Send shutdown sentinels and reap workers (destroying their connections)."""
        if self._closed:
            return
        self._closed = True
        for _ in self.processes:
            self.task_queue.put(None)
        for p in self.processes:
            p.join(timeout=15)
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()
        self.task_queue.close()
        self.result_queue.close()
        logger.info("Closed %d workers for db=%s", self.num_workers, self.db)


class PlanSearchEngine:
    """Per-workload engine: enumerates join orders and returns the best hint.

    Holds a persistent worker pool and a lazily-loaded DACE cost model.
    """

    def __init__(
        self,
        db: str,
        num_workers: Optional[int] = None,
        limit_num: int = 500_000,
        result_timeout: float = 120.0,
    ):
        self.db = db
        self.num_workers = num_workers if num_workers is not None else _default_num_workers()
        self.limit_num = limit_num
        self.result_timeout = result_timeout
        self._pool: Optional[WorkerPool] = None
        self._cost_model = None

    def _ensure_pool(self) -> WorkerPool:
        if self._pool is None:
            self._pool = WorkerPool(self.db, self.num_workers, self.result_timeout)
        return self._pool

    def _ensure_cost_model(self):
        if self._cost_model is None:
            # Imported lazily: pulls in torch, which must never touch workers.
            from plan_generator.cost_model import CostModel

            self._cost_model = CostModel()
        return self._cost_model

    def search(self, query: Query) -> Tuple[str, float]:
        """Return (best Leading-only pg_hint_plan comment, DACE inference time).

        The returned time is the sum of DACE model inference over the accepted
        physical plans only — consistent with the greedy strategy, which also
        reports model inference time. The (much larger) PostgreSQL EXPLAIN
        round-trips are deliberately NOT counted as inference time.
        """
        aliases, trees = enumerator.enumerate_join_orders(query, self.limit_num)
        num_plans = len(trees)
        logger.info("Query has %d join orders (limit_num=%d).", num_plans, self.limit_num)
        if num_plans == self.limit_num:
            logger.warning(
                "Join-order count hit the limit_num cap (%d): the C++ enumerator "
                "sampled instead of enumerating exhaustively. Raise --limit_num for "
                "a full search of this query.",
                self.limit_num,
            )

        if len(aliases) < 2 or not trees:
            return "", 0.0

        sql = enumerator.build_subquery_sql(query)
        leading_list = [enumerator.leading_from_tree(t, aliases) for t in trees]
        tasks = [(i, sql, leading) for i, leading in enumerate(leading_list)]

        start = time.perf_counter()
        pool = self._ensure_pool()
        infer_time = 0.0
        accepted = 0
        rejected = 0
        best_cost = float("inf")
        best_leading: Optional[str] = None
        try:
            cost_model = self._ensure_cost_model()
            # Buffer accepted plans and run DACE inference in batches (one
            # forward pass per chunk instead of one per plan), keeping memory
            # bounded while workers keep streaming results.
            chunk_size = 1024
            pending_ids: List[int] = []
            pending_plans: List[dict] = []

            def flush():
                nonlocal infer_time, best_cost, best_leading
                t0 = time.perf_counter()
                costs = cost_model.estimate_cost_batch(pending_plans)
                infer_time += time.perf_counter() - t0
                for tid, cost in zip(pending_ids, costs):
                    if cost < best_cost:
                        best_cost = cost
                        best_leading = leading_list[tid]

            for task_id, plan in pool.iter_results(tasks):
                if plan is None:
                    rejected += 1
                    continue
                accepted += 1
                pending_ids.append(task_id)
                pending_plans.append(plan)
                if len(pending_plans) >= chunk_size:
                    flush()
                    pending_ids.clear()
                    pending_plans.clear()
            if pending_plans:
                flush()
        except WorkerPoolError:
            logger.exception("Worker pool died; query failed. Pool will be recreated.")
            self._pool = None  # old pool closed; next query starts fresh
            return "", 0.0

        elapsed = time.perf_counter() - start
        logger.info(
            "Search %d join orders: %d accepted, %d rejected, best cost %.3f, "
            "DACE infer %.3fs, wall %.2fs",
            len(tasks), accepted, rejected, best_cost, infer_time, elapsed,
        )
        if best_leading is None:
            return "", infer_time
        return enumerator.hint_from_leading(best_leading), infer_time

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
