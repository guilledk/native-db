from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)
import warnings

import msgspec
import polars as pl

from polars.datatypes import TemporalType

from native_db._utils import NativeDBWarning
from native_db.dtypes import Mono
from native_db.errors import InvalidTableLayoutError
from native_db.schema import Column, Schema

if TYPE_CHECKING:
    from native_db.table import Table


class BasePartition(msgspec.Struct, frozen=True):
    on_column: str | int = 0

    def column(self, schema: Schema) -> Column:
        if isinstance(self.on_column, int):
            col = schema.columns[self.on_column]

        else:
            col = next(
                (c for c in schema.columns if c.name == self.on_column), None
            )
            if not col:
                raise ValueError(
                    f'No column with name {self.on_column} found in {schema}'
                )

        return col

    def encode(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)

    @classmethod
    def convert(cls, obj: dict[str, Any]) -> Self:
        return msgspec.convert(obj, type=cls)

    def runtime_for(self, table: 'Table') -> 'Partitioner':
        raise NotImplementedError


@runtime_checkable
class Partitioner(Protocol):
    meta: BasePartition
    col: Column
    plcol: pl.Expr
    by_cols: list[str]

    def prepare(
        self,
        staged: pl.LazyFrame
    ) -> pl.LazyFrame:
        '''
        Return `staged` with *additional* columns that will be used as hive
        partition keys (must be deterministic, derived from data or options).
        Do not drop or rename user columns.
        '''
        ...


class RowLenPartition(BasePartition, tag='rowlen', frozen=True):
    row_size: int = 100_000

    def runtime_for(self, table: 'Table') -> Partitioner:
        return RowLenPartitioner(self, table)


class RowLenPartitioner(Partitioner):
    by_cols: list[str] = ['bucket']

    def __init__(self, meta: RowLenPartition, table: 'Table') -> None:
        self.meta = meta
        self._table = table
        self.col = meta.column(table.schema)
        self.plcol = pl.col(self.col.name)
        self._row_size = meta.row_size

    def prepare(self, staged: pl.LazyFrame) -> pl.LazyFrame:
        # map to buckets by fixed range size row_size
        return staged.with_columns(
            (self.plcol.len() // self._row_size).alias(self.by_cols[0])
        )



class MonoPartition(BasePartition, tag='mono', frozen=True):
    row_size: int = 100_000

    def runtime_for(self, table: 'Table') -> Partitioner:
        return MonoPartitioner(self, table)


class MonoPartitioner(Partitioner):
    by_cols: list[str] = ['bucket']

    def __init__(self, meta: MonoPartition, table: 'Table') -> None:
        self.meta = meta
        self._table = table
        self.col = meta.column(table.schema)
        self.plcol = pl.col(self.col.name)

        if not isinstance(self.col.type, Mono):
            raise InvalidTableLayoutError(
                'Mono partitioner expected column type to be monotonic'
            )

        self._row_size = meta.row_size

    def prepare(self, staged: pl.LazyFrame) -> pl.LazyFrame:
        # map to buckets by fixed range size row_size
        return staged.with_columns(
            (self.plcol // self._row_size).alias(self.by_cols[0])
        )


TimePartKind = Literal['year', 'month', 'day']


class TimePartition(BasePartition, tag='time', frozen=True):
    kind: TimePartKind = 'year'

    def runtime_for(self, table: 'Table') -> Partitioner:
        return TimePartitioner(self, table)


class TimePartitioner(Partitioner):
    def __init__(self, meta: TimePartition, table: 'Table') -> None:
        self.meta = meta
        self._table = table
        self.col = meta.column(table.schema)
        self.plcol = pl.col(self.col.name)

        if not isinstance(self.col.type, TemporalType):
            raise InvalidTableLayoutError(
                'Time partitioner expected column type to be temporal'
            )

        self.by_cols = [meta.kind]
        self._part_exp: pl.Expr = getattr(self.plcol.dt, meta.kind)().alias(meta.kind)

    def prepare(self, staged: pl.LazyFrame) -> pl.LazyFrame:
        return staged.with_columns(self._part_exp)


class DictionaryPartition(BasePartition, tag='dict', frozen=True):
    """
    Dictionary-style partitioner: derive N single-character keys from the
    column's *string representation*.

    - depth: how many characters/digits to expose as hive keys (left-to-right).
    - pad_to: if set and the input is numeric, left-pad with zeros to this width
              before slicing (stable MSD buckets across variable-length ints).
    - lowercase: for string-like inputs, optionally lowercase before slicing.
    - signed: for numeric inputs, include the sign ('-' or '+') in the string.
              If False, uses absolute value.
    """
    depth: int = 1
    pad_to: int | None = None
    lowercase: bool = False
    signed: bool = False

    def runtime_for(self, table: 'Table') -> 'Partitioner':
        return DictionaryPartitioner(self, table)


class DictionaryPartitioner(Partitioner):
    def __init__(self, meta: DictionaryPartition, table: 'Table') -> None:
        self.meta = meta
        self._table = table
        self.col = meta.column(table.schema)
        self.plcol = pl.col(self.col.name)

        # Decide mode by schema "kind" (string/numeric) which your Column provides.
        kind = self.col.kind  # 'string' | 'numeric' | 'nested'
        if kind not in ('string', 'numeric'):
            raise InvalidTableLayoutError(
                'Dictionary partitioner expects a string or numeric column'
            )

        # Name the derived hive keys (keep old char* naming for compatibility).
        self.by_cols = [f'char{i}' for i in range(meta.depth)]
        self._kind = kind

        # Build the expression that yields the string to slice.
        if kind == 'string':
            s = self.plcol.cast(pl.Utf8)
            if self.meta.lowercase:
                s = s.str.to_lowercase()
            self._string_expr = s

        else:
            # Numeric: format to base-10 text, with optional sign and zero-pad.
            # - signed=False -> abs(); signed=True -> keep sign (leading '-' or '+')
            n = self.plcol
            if not self.meta.signed:
                n = n.abs()

            s = n.cast(pl.Utf8)

            if self.meta.pad_to:
                s = s.str.zfill(self.meta.pad_to)

            else:
                warnings.warn(
                    'Using DictionaryPartitioner on numeric column without '
                    'pad_to setting, might produce unstable buckets on '
                    'variable-length ints',
                    NativeDBWarning,
                    stacklevel=4,
                )

            self._string_expr = s

    def prepare(self, staged: pl.LazyFrame) -> pl.LazyFrame:
        # Derive char0..char{depth-1} using left-to-right slicing (MSD first).
        derived = [
            self._string_expr.str.slice(i, 1).alias(name)
            for i, name in enumerate(self.by_cols)
        ]
        return staged.with_columns(*derived)


PartitionerTypes = (
    MonoPartitioner | TimePartitioner | DictionaryPartitioner
)

PartitionTypes = (
    RowLenPartition | MonoPartition | TimePartition | DictionaryPartition
)
