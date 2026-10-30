import json
import datetime
from typing import Any

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


@singleton
class AnyColumn:
    def __init__(self):
        self.placeholder = "$col"
        
    def __str__(self) -> str:
        return self.placeholder

    def name(self) -> str:
        return self.placeholder


@singleton
class AnyOperator:
    def __init__(self):
        self.placeholder = "$op"
        
    def __str__(self) -> str:
        return self.placeholder


@singleton
class AnyLiteral:
    def __init__(self):
        self.placeholder = "$val"
        
    def __str__(self) -> str:
        return self.placeholder


any_column = AnyColumn()
any_operator = AnyOperator()
any_literal = AnyLiteral()

def value_wrapper(value) -> str:
    if value is None:
        return "NULL"
    elif isinstance(value, datetime.datetime):
        return datetime.datetime.strftime(value, "'%Y-%m-%d %H:%M:%S'")
    elif isinstance(value, datetime.timedelta):
        return str(f"INTERVAL {int(value.total_seconds())} SECONDS")
    elif isinstance(value, str):
        return "'%s'" % value.replace("'", "''")
    else:
        return str(value)

def parse_value(value_str) -> Any:
    if value_str == "NULL":
        return None
    elif value_str.startswith("'") and value_str.endswith("'"):
        if ":" in value_str:
            try:
                return datetime.datetime.strptime(value_str, "'%Y-%m-%d %H:%M:%S'")
            except ValueError:
                return value_str[1:-1].replace("''", "'")
        return value_str[1:-1].replace("''", "'")
    elif value_str.startswith("INTERVAL"):
        seconds = int(value_str.split()[1])
        return datetime.timedelta(seconds=seconds)
    elif value_str == str(any_literal):
        return any_literal
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str

def date_to_int(date: datetime.date) -> int:
    return int((date - datetime.datetime(1970, 1, 1)).total_seconds())
