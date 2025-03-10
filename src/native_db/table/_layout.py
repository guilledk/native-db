from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import warnings
import msgspec

from native_db._utils import NativeDBWarning
from native_db.errors import InvalidTableLayoutError
from native_db.schema import Column, Schema
from native_db.dtypes import Mono32, Mono64


@runtime_checkable
class PartOptionsProto(Protocol):
    on_column: str | int

    def column(self, schema: Schema) -> Column:
        ...

    def partition(self, base_path: Path, key: Any) -> Path:
        ...


class PartitionOptions(msgspec.Struct, frozen=True):
    on_column: str | int

    @lru_cache
    def column(self, schema: Schema) -> Column:
        if isinstance(self.on_column, int):
            col = schema.columns[self.on_column]

        else:
            col = next((
                c
                for c in schema.columns
                if c.name == self.on_column
            ), None)
            if not col:
                raise ValueError(f'No column with name {self.on_column} found in {schema}')

        return col


class MonoPartOptions(PartitionOptions, frozen=True):
    row_size: int
    zpad_size: int = 8

    def __post_init__(self) -> None:
        if self.row_size <= 0:
            raise InvalidTableLayoutError('MonoPartOptions\'s row_size must be >= 0')

    def partition(self, base_path: Path, key: Any) -> Path:
        assert isinstance(key, int)
        pid = key // self.row_size
        return base_path / str(pid).zfill(self.zpad_size)


class TimePartOptions(PartitionOptions, frozen=True):
    delta: timedelta

    def __post_init__(self) -> None:
        if self.delta.total_seconds() == 0:
            raise InvalidTableLayoutError('TimePartOptions\'s delta must be non-zero')

    def partition(self, _: Path, key: Any) -> Path:
        assert isinstance(key, datetime)
        # TODO: Impl
        raise NotImplementedError


PartOptionsTypes = MonoPartOptions | TimePartOptions


class LayoutOptions(msgspec.Struct, frozen=True):
    rows_per_file: int = 64_000
    start_frame_index: int = 0
    commit_threshold: int = 64_000 * 4

    partitioning: PartOptionsTypes | type[PartOptionsProto] | None = None

    def validate_for(self, schema: Schema) -> None:
        if self.commit_threshold % self.rows_per_file != 0:
            warnings.warn(
                'LayoutOptions\'s commit_threshold'
                f'({self.commit_threshold}) is not divisible by rows_per_file '
                f'({self.rows_per_file}); commits may need to leave rows on '
                'staging area until next commit or finish is called.',
                NativeDBWarning,
                stacklevel=2,
            )

        if not self.partitioning:
            return

        part_opts = self.partitioning
        if not isinstance(part_opts, PartOptionsProto):
            raise InvalidTableLayoutError(
                'Provided partitioning options not following PartOptionsProto'
            )

        match part_opts:
            case MonoPartOptions():
                if part_opts.row_size % self.commit_threshold != 0:
                    raise InvalidTableLayoutError(
                        'MonoPartOptions\'s row_size must be divisible by LayoutOptions\'s commit_threshold'
                    )

                col = part_opts.column(schema)

                if type(col.type) not in (Mono32, Mono64):
                    raise TypeError('Only monotonic types allowed on partition columns')

            case TimePartOptions():
                raise NotImplementedError
