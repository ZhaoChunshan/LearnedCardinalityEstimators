"""Data structures for query physical plan trees."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Union


class ScanMethod(Enum):
    """How we scan a table. Pick one."""
    SeqScan = "SeqScan"
    IndexScan = "IndexScan"
    IndexOnlyScan = "IndexOnlyScan"
    BitmapScan = "BitmapScan"


class JoinMethod(Enum):
    """How we join two relations. Pick one."""
    NestedLoopJoin = "NestLoop"
    HashJoin = "HashJoin"
    MergeJoin = "MergeJoin"


@dataclass
class ScanNode:
    """Leaf node. Reads from a table."""
    alias: str
    scan_method: ScanMethod = ScanMethod.SeqScan
    step_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class JoinNode:
    """Internal node. Joins two children."""
    join_method: JoinMethod = JoinMethod.NestedLoopJoin
    left_child: Optional[Union["JoinNode", ScanNode]] = None
    right_child: Optional[Union["JoinNode", ScanNode]] = None
    step_cost: float = 0.0
    total_cost: float = 0.0
