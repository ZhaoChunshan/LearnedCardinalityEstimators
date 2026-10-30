from enum import Enum
from typing import List, Dict, Callable, Any


class ColumnType(Enum):
    CATEGORICAL = 1
    NUMERICAL = 2
    STRING_WITH_CMP = 3
    OTHER = 4


class OperatorType(Enum):
    EQU = 1     # = 
    CMP = 2     # < > <= >=


def eval_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    return a == b

def eval_less_than(a, b) -> bool:
    if a is None or b is None:
        return False
    return a < b

def eval_greater_than(a, b) -> bool:
    if a is None or b is None:
        return False
    return a > b

def eval_less_than_or_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    return a <= b

def eval_greater_than_or_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    return a >= b


OPERATORS: Dict[OperatorType, List[str]] = {
    OperatorType.EQU: ["="],
    OperatorType.CMP: ["<", ">", "<=", ">=", (">", "<"), (">", "<="), (">=", "<"), (">=", "<=")],
}

OPERATOR_TYPE_PROB: Dict[OperatorType, float] = {
    OperatorType.EQU: 0.2,
    OperatorType.CMP: 0.8
}

COLUMN_OPERATOR_TYPE: Dict[ColumnType, List[OperatorType]] = {
    ColumnType.CATEGORICAL: [OperatorType.EQU],
    ColumnType.NUMERICAL: [OperatorType.EQU, OperatorType.CMP],
    ColumnType.STRING_WITH_CMP: [OperatorType.EQU, OperatorType.CMP],
}

OPERATOR_CMP_FUNC: Dict[str, Callable[[Any, Any], bool]] = {
    "=": eval_equal,
    "<": eval_less_than,
    ">": eval_greater_than,
    "<=": eval_less_than_or_equal,
    ">=": eval_greater_than_or_equal
}

OPERATOR_ID: Dict[str, int] = {
    "=": 0,
    "<": 1,
    ">": 2,
    "<=": 3,
    ">=": 4
}
