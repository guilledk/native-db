import fnmatch
import os
import sys
import time
import logging

from typing import Any, AsyncGenerator, Iterable, Sequence
from logging import Logger
from pathlib import Path
from itertools import count
from contextlib import asynccontextmanager

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

import polars as pl

from native_db.lowlevel import PolarsExecutor
from native_db.lowlevel.diskops import sink_frame
from native_db.structs import Struct
from native_db.table import Table, TableLike
from native_db.table.builder import TableBuilder
from native_db.table.writer import StagedPart
from native_db.transform import Transform


log = logging.getLogger(__name__)


class Context:
    def __init__(
        self,
        tables: Sequence[TableLike],
        *,
        datadir: str | Path | None = None,
        transforms: Sequence[Transform],
        log: Logger = log
    ) -> None:
        self.tables: tuple[Table, ...] = tuple(
            Table.from_like(t) for t in tables
        )
        if datadir:
            self.tables = tuple(
                t.copy(datadir=Path(datadir)) for t in self.tables
            )
        self.transforms: tuple[Transform, ...] = tuple(transforms)

        self._table_map: dict[str, Table] = {t.name: t for t in self.tables}
        self._tf_map: dict[str, Transform] = {tf.name: tf for tf in self.transforms}

        self._log = log

    def __getattr__(self, name: str) -> Table | Transform:
        '''
        If a normal attribute wasn't found, try resolving it as a table or a transform.

        '''
        try:
            return self._table_map[name]

        except KeyError:
            try:
                return self._tf_map[name]
            except KeyError:
                # important: raise AttributeError for correct Python semantics
                raise AttributeError(
                    f"{type(self).__name__!s} has no attribute {name!r} "
                    f"(and no table/transform with that name)"
                ) from None

    def __getitem__(self, name: str):
        '''
        Allow ctx['table_name'] or ctx['transform_name'].

        '''
        if name in self._table_map:
            return self._table_map[name]
        if name in self._tf_map:
            return self._tf_map[name]
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return name in self._table_map or name in self._tf_map

    def __dir__(self) -> list[str]:
        '''
        Make tab-completion show table/transform names.

        '''
        base = super().__dir__()
        return sorted(set(base) | set(self._table_map) | set(self._tf_map))

    def ensure_cache(
        self,
        regen: bool = False,
        skip_remotes: bool = False,
    ) -> None:
        '''
        Ensure all transform & remote caches are present on disk, generate if missing,
        or regenerate all if `regen` == True.

        '''
        self._log.info(f'ensuring table caches...')
        start = time.perf_counter()

        for table in self.tables:
            if table.src_kind == 'remote' and skip_remotes:
                continue

            self._log.info(
                f'generating cache for {table.name} at {table.local_path}...'
            )

            inner_start = time.perf_counter_ns()

            # materialize and return frame length
            table._frame = None
            try:
                # materialize and return frame length
                row_count: int = table.scan().select(pl.len()).collect().item()

            except pl.exceptions.ComputeError as e:
                self._log.info(f'cache generation for {table.name} failed, compute error "{e}"... skip.')
                continue

            # start calc with milliseconds
            elapsed = (
                time.perf_counter_ns() - inner_start
            ) // 1_000_000

            # upgrade to seconds
            if elapsed >= 1_000:
                elapsed /= 1_000
                elapsed_str = f'{elapsed:,.2f} sec'

            else:
                elapsed_str = f'{elapsed:,} ms'

            self._log.info(
                f'table cache for {table.name} generated, '
                f'took {elapsed_str}, row count: {row_count:,}, size: '
                f'{table.disk_size:,} bytes'
            )

        total_elapsed_sec = time.perf_counter() - start
        self._log.info(f'table caches ensured, took {total_elapsed_sec:,.2f} seconds')

        if self.transforms:
            self._log.info(f'ensuring transform caches... this might take a minute...')
            start = time.perf_counter()
            for name, transform in self._tf_map.items():
                if transform.is_cached:
                    if not regen:
                        self._log.info(
                            f'transform cache for {name} found at '
                            f'{transform.cache_path}'
                        )
                        continue

                    else:
                        transform.clear_cache()

                if transform.is_cached:
                    continue

                self._log.info(
                    f'generating transform cache for {name} at '
                    f'{transform.cache_path}...'
                )

                inner_start = time.perf_counter_ns()

                try:
                    # materialize and return frame length
                    row_count: int = transform.scan(ctx=self).select(pl.len()).collect().item()

                except pl.exceptions.ComputeError as e:
                    self._log.info(f'transform generation failed, compute error "{e}"... skip.')
                    continue

                except FileNotFoundError as e:
                    self._log.info(f'transform generation failed, "{e}"')
                    continue

                # start calc with milliseconds
                elapsed = (
                    time.perf_counter_ns() - inner_start
                ) // 1_000_000

                # upgrade to seconds
                if elapsed >= 1_000:
                    elapsed /= 1_000
                    elapsed_str = f'{elapsed:,.2f} sec'

                else:
                    elapsed_str = f'{elapsed:,} ms'

                self._log.info(
                    f'transform cache for {name} generated, '
                    f'took {elapsed_str}, row count: {row_count:,}, size: '
                    f'{transform.disk_size:,} bytes'
                )

            total_elapsed_sec = time.perf_counter() - start
            self._log.info(f'transform caches ensured, took {total_elapsed_sec:,.2f} seconds')


class ContextBuilder:
    '''
    Small helper to drive all the `TableBuilder` instances of a dataset all at
    once.

    '''

    def __init__(
        self,
        ctx: Context,
        *,
        executor: PolarsExecutor = PolarsExecutor()
    ):
        self._ctx = ctx
        self._executor = executor
        self._builders: dict[str, TableBuilder] = {}
        self._stage_threshold: dict[str, int | None] = {}
        # per-table staging root and commit sequence
        self._staging_root: dict[str, Path] = {}
        self._commit_id: dict[str, count[int]] = {}

        for table in ctx.tables:
            if table.src_kind != 'local':
                continue
            thr = table.writer_opts.stage_threshold if table.writer_opts else None
            self._builders[table.name] = table.builder(target_row_size=thr)
            self._stage_threshold[table.name] = thr
            root = getattr(self._ctx, table.name).local_path / '.staging' / f'pid-{os.getpid()}'
            root.mkdir(parents=True, exist_ok=True)
            self._staging_root[table.name] = root
            self._commit_id[table.name] = count(0)

    def append(self, table: str, row: Iterable[Any]) -> None:
        self._builders[table].append(row)

    def extend(self, table: str, rows: Iterable[Iterable[Any]]) -> None:
        self._builders[table].extend(rows)

    def push(self, row_sets: Iterable[dict[str, Iterable[Iterable[Any]]]]) -> None:
        for rows in row_sets:
            for table, row_iter in rows.items():
                self._builders[table].extend(row_iter)

    def flush_frames(self, *, allow_underfilled: bool = False, drain: bool = False) -> dict[str, pl.DataFrame]:
        return {
            name: builder.flush_frame(allow_underfilled=allow_underfilled, drain=drain)
            for name, builder in self._builders.items()
        }

    async def stage(
        self,
        *,
        drain: bool = False,
    ) -> dict[str, list[StagedPart]]:
        """
        Materialize per-table commits under `.staging/pid-<PID>/commit-<N>/...`
        writing *partitioned* files ready for integration. No frame indices.
        """
        staged: dict[str, list[StagedPart]] = {}
        ops: list[pl.LazyFrame] = []
        to_finalize: list[tuple[str, Path, Path]] = []  # (table_name, tmp_dir, final_dir)

        for table_name, builder in self._builders.items():
            table = getattr(self._ctx, table_name)
            target = self._stage_threshold.get(table_name)
            root = self._staging_root[table_name]

            def _prepare_and_queue(df: pl.DataFrame) -> None:
                commit_id = next(self._commit_id[table_name])
                tmp = root / f'commit-{commit_id}.tmp'
                final = root / f'commit-{commit_id}'
                tmp.mkdir(parents=True, exist_ok=True)

                # Add partition keys if needed and sink partitioned.
                part = table.partitioning
                lf = df.lazy()
                if part:
                    lf = part.prepare(lf)  # derive hive keys from data (e.g. bucket/year...)
                    # Preserve hinted sorts inside each partition.
                    sort_cols = [
                        pl.col(col.name)
                        for col in table.schema.columns
                        if col.hints.sort in ("asc", "desc")
                    ]
                    scheme = pl.PartitionByKey(
                        tmp,
                        by=part.by_cols,
                        include_key=False,
                        per_partition_sort_by=sort_cols or [],
                    )
                    ops.append(sink_frame(lf, scheme, **table.sink_args()))
                else:
                    # Single-file, unpartitioned commit
                    out = tmp / f"part.{table.format}"
                    ops.append(sink_frame(lf, out, **table.sink_args()))

                to_finalize.append((table_name, tmp, final))

            if target is None:
                df = builder.flush_frame(allow_underfilled=False, drain=drain)
                if df.height > 0:
                    _prepare_and_queue(df)
                continue

            # Emit as many exact-size chunks as available.
            while builder.rows() >= target:
                df = builder.flush_frame()
                _prepare_and_queue(df)

            # On drain, flush final underfilled remainder (once).
            if drain and builder.rows() > 0:
                df = builder.flush_frame(drain=True)
                if df.height > 0:
                    _prepare_and_queue(df)

        if not ops:
            return {}

        # Execute all sinks, then flip tmp -> final atomically per-commit.
        await self._executor.collect_all(ops)
        for _table_name, tmp, final in to_finalize:
            os.replace(tmp, final)

        # Walk commits and emit StagedPart records for the writer(s).
        for table_name, _tmp, commit_root in to_finalize:
            table = getattr(self._ctx, table_name)
            parts: list[StagedPart] = []
            for dirpath, _dirs, files in os.walk(commit_root):
                for name in files:
                    if fnmatch.fnmatch(name, table.file_pattern):
                        p = Path(dirpath) / name
                        rel_parent = Path(dirpath).relative_to(commit_root)
                        parts.append(
                            StagedPart(
                                path=p,
                                rel_parent=rel_parent,
                                commit_root=commit_root,
                                format=table.format,
                            )
                        )
            if parts:
                staged.setdefault(table_name, []).extend(parts)

        return staged



class _WriterState(Struct):
    table: 'Table'
    send_chan: MemoryObjectSendStream[StagedPart]
    recv_chan: MemoryObjectReceiveStream[StagedPart]


def _excepthook(exc_type, exc, tb):
    import traceback
    tbe = traceback.TracebackException.from_exception(
        exc, max_group_depth=1_000, max_group_width=1_000
    )
    sys.stderr.write(''.join(tbe.format()))

sys.excepthook = _excepthook


class ContextWriter:
    '''
    Small helper to drive all the `TableWriter` instances of a dataset all at
    once.

    '''

    def __init__(
        self,
        ctx: Context,
        tg: anyio.abc.TaskGroup,
        *,
        log: Logger | None = None,
        executor: PolarsExecutor | None = None
    ) -> None:
        self.ctx = ctx
        self._tg = tg
        self._log = log
        self._executor = executor
        self._stage_limit = anyio.CapacityLimiter(3)

        self._writers: dict[str, _WriterState] = {}
        for table in self.ctx.tables:
            if table.src_kind != 'local':
                continue

            schan, rchan = anyio.create_memory_object_stream[StagedPart](1)
            self._writers[table.name] = _WriterState(
                table=table,
                send_chan=schan,
                recv_chan=rchan,
            )

        for table in self._writers:
            tg.start_soon(self._table_proxy, table)

    def _ensure_table(self, table: str) -> _WriterState:
        w = self._writers.get(table)
        if w is None:
            raise RuntimeError(f'Unknown table: {table}')

        return w

    async def _table_proxy(self, table: str) -> None:
        state = self._ensure_table(table)
        w = state.table.writer(executor=self._executor)
        try:
            async with state.recv_chan:
                async for frame in state.recv_chan:
                    await w.stage_direct(frame)  # normal path

        finally:
            with anyio.CancelScope(shield=True):
                # channel closed: finalize
                await w.drain()  # flush leftovers
                log.info('drained')

    async def aclose(self) -> None:
        for wstate in self._writers.values():
            await wstate.send_chan.aclose()

    async def stage_direct(self, name: str, frame: StagedPart) -> None:
        state = self._ensure_table(name)
        await state.send_chan.send(frame)

    async def stage_all(
        self,
        row_sets: dict[str, list[StagedPart]]
    ) -> None:
        async with self._stage_limit:
            for name, frames in row_sets.items():
                for frame in frames:
                    self._tg.start_soon(self.stage_direct, name, frame)

@asynccontextmanager
async def open_ctx_writer(
    ctx: Context,
    *,
    executor: PolarsExecutor | None = None
) -> AsyncGenerator[ContextWriter, None]:
    async with anyio.create_task_group() as tg:
        w = ContextWriter(
            ctx,
            tg,
            executor=executor
        )
        yield w
        await w.aclose()
