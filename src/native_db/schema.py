from __future__ import annotations

from functools import cached_property
from inspect import isclass
from typing import Iterable

import msgspec
import polars as pl

from native_db.dtypes import (
    DataTypeExt,
    DataTypeMeta,
    TypeHints,
    TypeKind,
    avg_type_size,
    is_custom_type,
    py_type_for,
    should_use_dictionary,
    type_kind,
)
from native_db.structs import FrozenStruct


class ColumnMeta(FrozenStruct, frozen=True):
    name: str
    type: DataTypeMeta
    hints: TypeHints


class Column(msgspec.Struct, dict=True, frozen=True):
    name: str
    type: DataTypeExt
    hints: TypeHints = TypeHints()

    @staticmethod
    def from_like(c: ColumnLike) -> Column:
        match c:
            case tuple():
                hints = TypeHints()
                name = c[0]
                typ = c[1]
                hints = c[2] if len(c) == 3 else TypeHints()
                return Column(name, typ, hints)

            case dict() | ColumnMeta():
                if isinstance(c, dict):
                    c = ColumnMeta.convert(c)

                return Column(c.name, c.type.decode(), c.hints)

        return c

    @property
    def avg_size(self) -> int:
        return avg_type_size(self.type, self.hints)

    @property
    def use_dictionary(self) -> bool:
        return self.hints.use_dictionary or (
            self.hints.use_dictionary is None
            and should_use_dictionary(self.type)
        )

    @property
    def kind(self) -> TypeKind:
        return type_kind(self.type)

    @property
    def py_type(self) -> type:
        return py_type_for(self.type, hints=self.hints)

    def encode(self) -> ColumnMeta:
        return ColumnMeta(
            name=self.name,
            type=DataTypeMeta.from_dtype(self.type),
            hints=self.hints,
        )


ColumnLike = (
    tuple[str, DataTypeExt]
    | tuple[str, DataTypeExt, TypeHints]
    | dict
    | ColumnMeta
    | Column
)


class SchemaMeta(FrozenStruct, frozen=True):
    columns: list[ColumnMeta]
    row_group_size: int


class Schema:
    def __init__(
        self,
        columns: Iterable[ColumnLike],
        *,
        row_group_size: int | None = None,
        target_group_size: int = 64 * 1024 * 1024,
    ):
        self._columns: tuple[Column, ...] = tuple(
            (Column.from_like(c) for c in columns)
        )

        self._avg_row_size = sum((col.avg_size for col in self._columns))

        if row_group_size:
            self._row_group_size = row_group_size

        else:
            if target_group_size <= 0:
                raise ValueError(f'target_group_size needs to be > 0')

            self._row_group_size = max(
                1, target_group_size // self.avg_row_size
            )

    @staticmethod
    def from_like(s: SchemaLike) -> Schema:
        match s:
            case Schema():
                return s

            case dict() | SchemaMeta():
                if isinstance(s, dict):
                    s = SchemaMeta.convert(s)

                return Schema(s.columns, row_group_size=s.row_group_size)

        return Schema(s)

    def __len__(self) -> int:
        return len(self._columns)

    def as_polars(self) -> pl.Schema:
        return pl.Schema(
            (
                col.name,
                col.type
                if not is_custom_type(col.type)
                else col.type.fallback_type(),
            )
            for col in self._columns
        )

    @cached_property
    def row_type(self) -> type:
        return tuple[tuple((col.py_type for col in self._columns))]

    @property
    def avg_row_size(self) -> int:
        return self._avg_row_size

    @property
    def row_group_size(self) -> int:
        return self._row_group_size

    @cached_property
    def columns(self) -> tuple[Column, ...]:
        return self._columns

    @cached_property
    def numeric_columns(self) -> tuple[Column, ...]:
        return tuple((col for col in self._columns if col.kind == 'numeric'))

    @cached_property
    def string_columns(self) -> tuple[Column, ...]:
        return tuple((col for col in self._columns if col.kind == 'string'))

    @cached_property
    def nested_columns(self) -> tuple[Column, ...]:
        return tuple((col for col in self._columns if col.kind == 'nested'))

    def pretty_str(self) -> str:
        '''Return a human-readable representation of the schema.'''
        lines = ['Schema:']
        for col in self._columns:
            hints = []
            if col.hints.optional:
                hints.append('optional')
            if col.hints.unique:
                hints.append('unique')
            if col.hints.sort != 'none':
                hints.append(f'sort={col.hints.sort}')
            if col.use_dictionary:
                hints.append('dict')
            hints.append(f'avg size: {col.avg_size}')
            hints_str = f' ({", ".join(hints)})' if hints else ''
            name = col.type.__name__ if isclass(col.type) else str(col.type)
            lines.append(f'  - {col.name}: {name}{hints_str}')
        lines.append(f'Row group size: {self.row_group_size:,}')
        lines.append(f'Average row size: {self.avg_row_size:,} bytes')
        return '\n'.join(lines)

    def encode(self) -> SchemaMeta:
        return SchemaMeta(
            columns=[col.encode() for col in self._columns],
            row_group_size=self._row_group_size,
        )


SchemaLike = Iterable[ColumnLike] | dict | SchemaMeta | Schema
