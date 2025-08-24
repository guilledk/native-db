from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import msgspec

from polars._typing import ParquetCompression

from native_db._utils import (
    fetch_remote_file,
    get_root_datadir,
    path_size,
    remote_src_protos,
)
from native_db.schema import Schema, SchemaLike, SchemaMeta
from native_db.table._layout import (
    MonoPartitionMeta,
    Partitioner as Partitioner,
    PartitionMetaTypes as PartitionMetaTypes,
    MonoPartitioner as MonoPartitioner,
)


class TableMeta(msgspec.Struct, frozen=True):
    name: str
    source: str
    schema: SchemaMeta
    compression: ParquetCompression
    compression_level: int | None
    prefix: str
    suffix: str
    partitioning: dict | None = None


class Table:
    def __init__(
        self,
        name: str,
        source: str | Path,
        schema: SchemaLike,
        *,
        compression: ParquetCompression = 'zstd',
        compression_level: int | None = None,
        prefix: str = 'rows',
        suffix: str = 'parquet',
        datadir: Path | None = None,
        partitioning: PartitionMetaTypes | dict = MonoPartitionMeta(),
    ) -> None:
        self.name = name
        self.schema = Schema.from_like(schema)
        self.source = source
        self.compression: ParquetCompression = compression
        self.compression_level = compression_level

        self._datadir = (datadir if datadir else get_root_datadir()).resolve()

        self._local_path: Path

        if isinstance(source, str):
            if any(
                (
                    source.startswith(f'{proto}://')
                    for proto in remote_src_protos
                )
            ):
                if source.startswith('http'):
                    self.source, self._local_path = fetch_remote_file(
                        self._datadir, source, prefix=prefix, suffix=suffix
                    )

                else:
                    raise RuntimeError(
                        f'TODO: implement remote source fetch for {source}'
                    )

            else:
                self._local_path = Path(source)

        elif isinstance(source, Path):
            self._local_path = source

        if not self._local_path.is_absolute():
            self._local_path = self._datadir / self._local_path

        self.prefix = prefix
        self.suffix = suffix

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

        part: PartitionMetaTypes = (
            partitioning
            if isinstance(partitioning, PartitionMetaTypes)
            else msgspec.convert(partitioning, type=PartitionMetaTypes)
        )

        self.partitioning: Partitioner = part.runtime_for(self)

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

    @cached_property
    def struct(self) -> type[msgspec.Struct]:
        return msgspec.defstruct(
            f'{self.name.capitalize()}Row',
            fields=((col.name, col.py_type) for col in self.schema.columns),
            module='native_db.autogen',
        )

    @property
    def local_path(self) -> Path:
        return self._local_path

    @property
    def exists(self) -> bool:
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
            f'File pattern: {self.prefix}*.{self.suffix}',
            '',
            self.schema.pretty_str(),
        ]
        return '\n'.join(lines)

    def copy(
        self,
        name: str | None = None,
        source: str | None = None,
        schema: SchemaLike | None = None,
        compression: ParquetCompression | None = None,
        compression_level: int | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        datadir: Path | None = None,
        partitioning: PartitionMetaTypes | dict | None = None,
    ) -> Table:
        '''
        Helper for creating a new table definition from self, optionally
        overriding some properties.

        '''
        return Table(
            name=name or self.name,
            source=source or self.name,
            schema=schema or self.schema,
            compression=compression or self.compression,
            compression_level=compression_level or self.compression_level,
            prefix=prefix or self.prefix,
            suffix=suffix or self.suffix,
            datadir=datadir or self._datadir,
            partitioning=partitioning or self.partitioning.meta.encode(),
        )

    def encode(self) -> TableMeta:
        return TableMeta(
            name=self.name,
            source=str(self.source),
            schema=self.schema.encode(),
            compression=self.compression,
            compression_level=self.compression_level,
            prefix=self.prefix,
            suffix=self.suffix,
            partitioning=self.partitioning.meta.encode(),
        )

    def scan(self, *, partition_hints: dict[str, Any] = {}) -> None: ...
