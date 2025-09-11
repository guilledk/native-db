from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import msgspec
import polars as pl

from native_db._utils import (
    fetch_remote_file,
    get_root_datadir,
    path_size,
    remote_src_protos,
    solve_redirects,
)
from native_db.errors import NativeDBError
from native_db.lowlevel.diskops import FrameFormats, format_from_path, scan_frame
from native_db.schema import Schema, SchemaLike, SchemaMeta
from native_db.structs import FrozenStruct
from native_db.table._layout import (
    Partitioner as Partitioner,
    PartitionTypes as PartitionTypes,
    RowLenPartitioner as RowLenPartitioner,
    RowLenPartition as RowLenPartition,
    MonoPartitioner as MonoPartitioner,
    MonoPartition as MonoPartition,
    DictionaryPartitioner as DictionaryPartitioner,
    DictionaryPartition as DictionaryPartition,
    TimePartitioner as TimePartitioner,
    TimePartition as TimePartition
)
from native_db.table.builder import TableBuilder
from native_db.table.writer import TableWriter, TableWriterOptions


class TableMeta(FrozenStruct, frozen=True):
    name: str
    source: str
    schema: SchemaMeta
    format: FrameFormats
    compression: str
    compression_level: int | None
    prefix: str | None
    suffix: str | None
    datadir: str | None
    partitioning: dict | None
    writer_opts: TableWriterOptions | None


TableSrcKind = Literal['local', 'remote']


class Table:
    def __init__(
        self,
        name: str,
        source: str | Path,
        schema: SchemaLike,
        *,
        format: FrameFormats | None = None,
        compression: str = 'zstd',
        compression_level: int | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        datadir: str | Path | None = None,
        partitioning: PartitionTypes | dict | None = None,
        writer_opts: TableWriterOptions | None = None,
        struct: type[msgspec.Struct] | None = None,
    ) -> None:
        self.name = name
        self.schema = Schema.from_like(schema)
        self.source = source
        self.format: FrameFormats = format or format_from_path(source)
        self.compression: str = compression
        self.compression_level = compression_level

        self._datadir = (Path(datadir) if datadir else get_root_datadir()).resolve()

        self._local_path: Path

        self.src_kind: str
        if isinstance(source, str):
            if any(
                (
                    source.startswith(f'{proto}://')
                    for proto in remote_src_protos
                )
            ):
                self.src_kind = 'remote'
                self._local_path = self._datadir / f'remote/{name}'

            else:
                self.src_kind = 'local'
                self._local_path = Path(source)

        elif isinstance(source, Path):
            self._local_path = source

        if not self._local_path.is_absolute():
            self._local_path = self._datadir / self._local_path

        self.prefix = prefix
        self.suffix = suffix
        self._suffix = suffix if suffix else format

        if (
            self.compression != 'uncompressed'
            and self.compression_level is None
        ):
            self.compression_level = (
                10
                if (
                    len(self.schema.string_columns) / len(self.schema.columns)
                    > 0.5
                    or len(self.schema.nested_columns) > 0
                )
                else 6
            )

        self.partitioning: Partitioner | None = None
        if partitioning:
            part: PartitionTypes = (
                partitioning
                if isinstance(partitioning, PartitionTypes)
                else msgspec.convert(partitioning, type=PartitionTypes)
            )

            self.partitioning = part.runtime_for(self)

        self.struct = (
            struct
            if struct
            else msgspec.defstruct(
                f'{self.name.capitalize()}Row',
                fields=((col.name, col.py_type) for col in self.schema.columns),
                module='native_db.autogen',
            )
        )
        self.writer_opts = writer_opts

        # used when calling .scan with use_cache=True
        self._frame: pl.LazyFrame | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented

        return (
            self.name == other.name
            and self.local_path == other.local_path
            and self.schema == other.schema
        )

    def __hash__(self) -> int:
        return hash((
            self.name,
            self.local_path,
            self.schema,
            self.partitioning,
        ))

    @staticmethod
    def from_like(t: TableLike, **kwargs) -> Table:
        if isinstance(t, Table):
            return t

        if isinstance(t, dict):
            t = TableMeta.convert(t)

        return Table(**t.to_dict(), **kwargs)

    @property
    def file_pattern(self) -> str:
        p = f'*.{self._suffix}'
        if self.prefix:
            p = '-'.join((self.prefix, p))

        return p

    @property
    def local_path(self) -> Path:
        if not isinstance(self._local_path, Path):
            raise NativeDBError(
                'Table internal _local_path not set? Likely remote table not downloaded'
            )

        return self._local_path

    @property
    def exists(self) -> bool:
        if not isinstance(self._local_path, Path):
            raise NativeDBError(
                'Table internal _local_path not set? Likely remote table not downloaded'
            )

        return self._local_path.exists()

    @property
    def disk_size(self) -> int:
        if not self.exists:
            raise FileNotFoundError(
                f'Called Table.disk_size but table not present'
            )

        return path_size(self.local_path)

    def pretty_str(self) -> str:
        '''Return a human-readable representation of the table metadata.'''
        lines = [
            f'Table: {self.name}',
            f'Path: {self.local_path}',
            f'Source: {self.source or "N/A"}',
            f'Compression: {self.compression} (level={self.compression_level})',
            f'File pattern: {self.prefix}*.{self._suffix}',
            '',
            self.schema.pretty_str(),
        ]
        return '\n'.join(lines)

    def copy(
        self,
        name: str | None = None,
        source: str | None = None,
        schema: SchemaLike | None = None,
        format: FrameFormats | None = None,
        compression: str | None = None,
        compression_level: int | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        datadir: Path | None = None,
        partitioning: PartitionTypes | dict | None = None,
        writer_opts: TableWriterOptions | None = None,
        struct: type[msgspec.Struct] | None = None
    ) -> Table:
        '''
        Helper for creating a new table definition from self, optionally
        overriding some properties.

        '''
        return Table(
            name=name or self.name,
            source=source or self.source,
            format=format or self.format,
            schema=schema or self.schema,
            compression=compression or self.compression,
            compression_level=compression_level or self.compression_level,
            prefix=prefix or self.prefix,
            suffix=suffix or self.suffix,
            datadir=datadir or self._datadir,
            partitioning=partitioning or (self.partitioning.meta.encode() if self.partitioning else None),
            writer_opts=writer_opts or self.writer_opts,
            struct=struct or self.struct
        )

    def encode(self) -> TableMeta:
        return TableMeta(
            name=self.name,
            source=str(self.source),
            schema=self.schema.encode(),
            format=self.format,
            compression=self.compression,
            compression_level=self.compression_level,
            prefix=self.prefix,
            suffix=self.suffix,
            datadir=str(self._datadir),
            partitioning=self.partitioning.meta.encode() if self.partitioning else None,
            writer_opts=self.writer_opts
        )

    def builder(self, target_row_size: int | None = None) -> TableBuilder:
        return TableBuilder(self, target_row_size=target_row_size)

    def writer(self, **kwargs) -> TableWriter:
        return TableWriter(
            self,
            options=self.writer_opts,
            **kwargs
        )

    def empty(self) -> pl.LazyFrame:
        return pl.LazyFrame(schema=self.schema.as_polars())

    def files(self) -> tuple[Path, ...]:
        if not self._local_path.exists():
            return tuple()

        return tuple(
            sorted(
                p
                for p in self._local_path.rglob(f'**/{self.file_pattern}')
                if '.staging' not in p.parts
            )
        )

    @property
    def disk_size(self) -> int:
        return sum(
            f.stat().st_size for f in self.files()
        )

    def part_file(self, i: int | str, *, zpad: int = 5) -> str:
        if isinstance(i, int):
            i = f'{i:0{zpad}d}'

        p = f'{i}.{self._suffix}'
        if self.prefix:
            p = '-'.join((self.prefix, p))
        return p

    def sink_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            'format': self.format,
        }
        if self.format != 'csv':
            args['compression'] = self.compression

        if self.format == 'parquet':
            args['compression_level'] = self.compression_level
            args['row_group_size'] = self.schema.row_group_size

        return args

    def scan_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            'format': self.format,
        }
        if self.format != 'csv':
            args['hive_partitioning'] = self.partitioning is not None
            # args['include_file_paths'] = True

        return args

    def scan(self, *, use_cache: bool = True) -> pl.LazyFrame:
        if use_cache and self._frame is not None:
            return self._frame

        try:
            if self._local_path and self._local_path.exists():
                frame = scan_frame(
                    self._local_path,
                    **self.scan_args(),
                )

            else:
                match self.src_kind:
                    case 'local':
                        frame = self.empty()

                    case 'remote':
                        assert isinstance(self.source, str)

                        if self.source.startswith('http'):
                            self.source = solve_redirects(self.source)
                            self._local_path = fetch_remote_file(
                                self._local_path, self.source,
                                prefix=self.prefix, suffix=self.suffix
                            )
                            frame = scan_frame(
                                self._local_path,
                                **self.scan_args(),
                            )

                        else:
                            raise NotImplementedError

                    case _:
                        raise NotImplementedError

        except pl.exceptions.ComputeError:
            frame = self.empty()

        if use_cache:
            self._frame = frame
            

        return frame


TableLike = dict | TableMeta | Table
