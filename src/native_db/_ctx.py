import os
import sys
import time
import logging

from typing import Any, AsyncGenerator, Iterable, Sequence
from logging import Logger
from pathlib import Path
from itertools import count
from collections import deque
from contextlib import asynccontextmanager

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

import polars as pl

from native_db.lowlevel import PolarsExecutor
from native_db.lowlevel.diskops import FrameFormats, sink_frame
from native_db.structs import Struct
from native_db.table import Table, TableLike
from native_db.table.builder import TableBuilder
from native_db.table.writer import StagedFrame, TableWriterOptions
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
        self._idx_queues: dict[str, deque[tuple[int, FrameFormats, str]]] = {}
        self._auto_index: dict[str, count[int]] = {}

        for table in ctx.tables:
            if table.src_kind != 'local':
                continue
            thr = table.writer_opts.stage_threshold if table.writer_opts else None
            # hand the target to the TableBuilder (it will only flush exact slices)
            self._builders[table.name] = table.builder(target_row_size=thr)  # ← key change
            self._stage_threshold[table.name] = thr
            self._idx_queues[table.name] = deque()
            self._auto_index[table.name] = count(0)

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
        frame_index: int | None = None,
        format: FrameFormats = 'ipc',
        compression: str = 'zstd',
        allow_underfilled: bool = False,
        drain: bool = False,
    ) -> dict[str, list[StagedFrame]]:
        """
        Queue the provided frame_index for every table; emit only *exactly*
        sized frames (if a per-table stage_threshold is configured), preserving
        the user's original index order per table.

        New:
          - allow_underfilled: if True, underfilled builders are a no-op (keeps order, writes nothing).
          - drain: if True, after emitting all exact-size frames, also emit ONE final partial remainder.
        """
        staged: dict[str, list[StagedFrame]] = {}
        ops: list[pl.LazyFrame] = []
        to_rename: list[tuple[Path, Path]] = []

        for table_name, builder in self._builders.items():
            table = getattr(self._ctx, table_name)
            target = self._stage_threshold.get(table_name)  # None => no exact sizing

            # 1) queue index (per table; preserves user call order)
            idxq = self._idx_queues[table_name]
            idx = frame_index if frame_index is not None else next(self._auto_index[table_name])
            idxq.append((idx, format, compression))

            # 2) emit frames
            if target is None:
                # Legacy behavior: flush whatever we have once (0 rows => skip).
                # Flags are passed for API symmetry; they only matter when target is set.
                df = builder.flush_frame(allow_underfilled=allow_underfilled, drain=drain)
                if df.height == 0:
                    continue
                use_idx, fmt, comp = idxq.popleft()
                stage_path = table.local_path / '.staging' / f'frame-{use_idx:05d}.{df.height}-rows.{fmt}'
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(f'{stage_path.resolve()}.tmp')
                to_rename.append((stage_path, tmp))
                ops.append(sink_frame(df.lazy(), tmp, compression=comp))
                staged.setdefault(table_name, []).append(
                    StagedFrame(index=use_idx, number_of_rows=df.height, path=stage_path, format=fmt)
                )
                continue

            # exact-size path: only flush while we have enough rows *and* queued indices
            while builder.rows() >= target and idxq:
                df = builder.flush_frame()  # emits exactly `target` rows here
                use_idx, fmt, comp = idxq.popleft()
                stage_path = table.local_path / '.staging' / f'frame-{use_idx:05d}.{df.height}-rows.{fmt}'
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(f'{stage_path.resolve()}.tmp')
                to_rename.append((stage_path, tmp))
                ops.append(sink_frame(df.lazy(), tmp, compression=comp))
                staged.setdefault(table_name, []).append(
                    StagedFrame(index=use_idx, number_of_rows=df.height, path=stage_path, format=fmt)
                )

            # If we're draining at end-of-stream, emit the underfilled remainder (once)
            if drain and builder.rows() > 0:
                df = builder.flush_frame(drain=True)  # remainder < target
                if df.height > 0:
                    if idxq:
                        use_idx, fmt, comp = idxq.popleft()
                    else:
                        # fallback: allocate a fresh index if none were queued for this table
                        use_idx = next(self._auto_index[table_name])
                        fmt, comp = format, compression
                    stage_path = table.local_path / '.staging' / f'frame-{use_idx:05d}.{df.height}-rows.{fmt}'
                    stage_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = Path(f'{stage_path.resolve()}.tmp')
                    to_rename.append((stage_path, tmp))
                    ops.append(sink_frame(df.lazy(), tmp, compression=comp))
                    staged.setdefault(table_name, []).append(
                        StagedFrame(index=use_idx, number_of_rows=df.height, path=stage_path, format=fmt)
                    )

        if not ops:
            # nothing to write this round
            return {}

        await self._executor.collect_all(ops)
        for final, tmp in to_rename:
            os.rename(tmp, final)

        # normalize value type for convenience (StagedFrame | list[StagedFrame])
        return {
            name: frames
            for name, frames in staged.items()
        }



class _WriterState(Struct):
    table: 'Table'
    send_chan: MemoryObjectSendStream[StagedFrame]
    recv_chan: MemoryObjectReceiveStream[StagedFrame]


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
        start_frame_index: int = 1,
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

            schan, rchan = anyio.create_memory_object_stream[StagedFrame](1)
            self._writers[table.name] = _WriterState(
                table=(
                    table.copy(
                        writer_opts=TableWriterOptions.from_other(
                            table.writer_opts,
                            start_frame_index=start_frame_index
                        )
                    )
                    if table.writer_opts else table
                ),
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

    async def stage_direct(self, name: str, frame: StagedFrame) -> None:
        state = self._ensure_table(name)
        await state.send_chan.send(frame)

    async def stage(
        self,
        name: str,
        path: str | Path,
        *,
        index: int | None = None,
        number_of_rows: int = 0,
        format: FrameFormats | None = None
    ) -> None:
        await self.stage_direct(
            name,
            StagedFrame.from_push(
                path=path,
                index=index,
                number_of_rows=number_of_rows,
                format=format
            )
        )

    async def stage_all(
        self,
        row_sets: dict[str, list[StagedFrame]]
    ) -> None:
        async with self._stage_limit:
            for name, frames in row_sets.items():
                for frame in frames:
                    self._tg.start_soon(self.stage_direct, name, frame)

    async def stage_frame(
        self,
        name: str,
        frame: pl.DataFrame,
        index: int,
        *,
        format: FrameFormats = 'ipc'
    ) -> None:
        table = getattr(self.ctx, name)
        path = table.local_path / '.staging' / f'frame-{index:05d}.{frame.height}.{format}'

        sinkop = sink_frame(
            frame.lazy(),
            path,
            format=format
        )

        if self._executor:
            await self._executor.collect(sinkop)

        else:
            await sinkop.collect_async()

        await self.stage(
            name,
            path,
            index=index,
            number_of_rows=frame.height,
            format=format
        )

@asynccontextmanager
async def open_ctx_writer(
    ctx: Context,
    *,
    start_frame_index: int = 0,
    executor: PolarsExecutor | None = None
) -> AsyncGenerator[ContextWriter, None]:
    async with anyio.create_task_group() as tg:
        w = ContextWriter(
            ctx,
            tg,
            start_frame_index=start_frame_index,
            executor=executor
        )
        yield w
        await w.aclose()
