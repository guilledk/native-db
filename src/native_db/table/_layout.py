from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

import msgspec
import polars as pl

from polars.datatypes import TemporalType

from native_db.dtypes import Keyword
from native_db.errors import InvalidTableLayoutError
from native_db.schema import Column, Schema

if TYPE_CHECKING:
    from native_db.table import Table


class BasePartitionMeta(msgspec.Struct, frozen=True):
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
    meta: BasePartitionMeta
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


class MonoPartitionMeta(BasePartitionMeta, tag='mono', frozen=True):
    row_size: int = 100_000

    def runtime_for(self, table: 'Table') -> Partitioner:
        return MonoPartitioner(self, table)


class MonoPartitioner(Partitioner):
    by_cols: list[str] = ['bucket']

    def __init__(self, meta: MonoPartitionMeta, table: 'Table') -> None:
        self.meta = meta
        self._table = table
        self.col = meta.column(table.schema)
        self.plcol = pl.col(self.col.name)

        from native_db.dtypes import DataTypeMono

        if not isinstance(self.col.type, DataTypeMono):
            raise InvalidTableLayoutError(
                'Mono partitioner expected column type to be monotonic'
            )

        self._row_size = meta.row_size
        self._allow_gaps = self.col.type.allow_gaps

    def prepare(self, staged: pl.LazyFrame) -> pl.LazyFrame:
        # map to buckets by fixed range size row_size
        return staged.with_columns(
            (self.plcol // self._row_size).alias(self.by_cols[0])
        )


TimePartKind = Literal['year', 'month', 'day']


class TimePartitionMeta(BasePartitionMeta, tag='time', frozen=True):
    kind: TimePartKind = 'year'

    def runtime_for(self, table: 'Table') -> Partitioner:
        return TimePartitioner(self, table)


class TimePartitioner(Partitioner):
    def __init__(self, meta: TimePartitionMeta, table: 'Table') -> None:
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


class KeywordPartitionMeta(BasePartitionMeta, tag='keyword', frozen=True):
    char_depth: int = 1

    def runtime_for(self, table: 'Table') -> 'Partitioner':
        return KeywordPartitioner(self, table)


class KeywordPartitioner(Partitioner):

    def __init__(self, meta: KeywordPartitionMeta, table: 'Table') -> None:
        self.meta = meta
        self._table = table
        self.col = meta.column(table.schema)
        self.plcol = pl.col(self.col.name)

        if self.col.type != Keyword:
            raise InvalidTableLayoutError(
                'Keyword partitioner expected column type to be keyword'
            )

        self.by_cols = [f'char{i}' for i in range(meta.char_depth)]

    def prepare(self, staged: pl.LazyFrame) -> pl.LazyFrame:
        """
        Adds derived columns: char0..char{depth-1}, each a 1-char slice at that index.
        Non-existent positions become empty strings.
        """
        # derive N single-char columns using .str.slice(offset, length=1)
        derived = [
            self.plcol.str.slice(i, 1).alias(name)
            for i, name in enumerate(self.by_cols)
        ]
        return staged.with_columns(*derived)


PartitionerTypes = MonoPartitioner | TimePartitioner | KeywordPartitioner
PartitionMetaTypes = MonoPartitionMeta | TimePartitionMeta | KeywordPartitionMeta
