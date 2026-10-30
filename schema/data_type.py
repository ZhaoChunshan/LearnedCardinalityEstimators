from enum import Enum
from datetime import date, datetime, timedelta


class DataType:
    def __init__(self, name: str, python_type: type) -> None:
        self._name = name
        self._python_type = python_type

    def name(self) -> str:
        return self._name

    def python_type(self) -> type:
        return self._python_type


_smallint = DataType("smallint", int)
_integer = DataType("integer", int)
_bigint = DataType("bigint", int)
_numeric = DataType("double precision", float)
_string = DataType("character varying", str)
_date = DataType("date", date)
_interval = DataType("interval", timedelta)


DATATYPES = {"bigint": _bigint,
             "integer": _integer,
             "smallint": _smallint,
             "double precision": _numeric,
             "character varying": _string,
             "date": _date,
             "interval": _interval}
