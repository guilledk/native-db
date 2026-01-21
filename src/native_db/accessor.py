from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from logging import Logger
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    TypeAlias,
    TypeVar,
)

import anyio
import polars as pl

from native_db._ctx import Context
from native_db.lowlevel import PolarsExecutor
from native_db.table._layout import DictionaryPartition, MonoPartition, TimePartition

TKey = TypeVar('TKey', bound=Hashable)
PartValues: TypeAlias = tuple[Any, ...]
PostFn: TypeAlias = Callable[[pl.LazyFrame], pl.LazyFrame]


# Partition indexing (planner)


class PartitionIndex(Generic[TKey]):
    '''
    Maps key -> hive-partition-values tuple matching `by_cols`.

    If by_cols == (), everything is in one group and no pruning happens.

    '''

    by_cols: tuple[str, ...]

    def __init__(self, by_cols: Sequence[str]) -> None:
        self.by_cols = tuple(by_cols)

    def part_values(self, key: TKey) -> PartValues:
        raise NotImplementedError

    def groups_for_keys(self, keys: Iterable[TKey]) -> dict[PartValues, set[TKey]]:
        out: dict[PartValues, set[TKey]] = {}
        for k in keys:
            pv = self.part_values(k)
            out.setdefault(pv, set()).add(k)
        return out

    def apply_partition_filter(self, lf: pl.LazyFrame, pv: PartValues) -> pl.LazyFrame:
        if not self.by_cols:
            return lf
        if len(pv) != len(self.by_cols):
            raise ValueError(f'Partition values mismatch: pv={pv}, by_cols={self.by_cols}')
        for col, val in zip(self.by_cols, pv, strict=True):
            lf = lf.filter(pl.col(col) == val)
        return lf


class NoPartitionIndex(PartitionIndex[TKey]):
    def __init__(self) -> None:
        super().__init__(by_cols=())

    def part_values(self, key: TKey) -> PartValues:
        return ()


class MonoBucketIndex(PartitionIndex[int]):
    def __init__(self, *, bucket_col: str = 'bucket', row_size: int) -> None:
        super().__init__((bucket_col,))
        self._row_size = int(row_size)

    def part_values(self, key: int) -> PartValues:
        return (int(key) // self._row_size,)


class DictPrefixIndex(PartitionIndex[Hashable]):
    def __init__(
        self,
        *,
        by_cols: Sequence[str],
        depth: int,
        pad_to: int | None,
        lowercase: bool,
        signed: bool,
    ) -> None:
        super().__init__(by_cols=by_cols)
        self._depth = int(depth)
        self._pad_to = pad_to
        self._lowercase = bool(lowercase)
        self._signed = bool(signed)

    def _to_prefix_string(self, key: Hashable) -> str:
        # Follow DictionaryPartitioner semantics as best as possible.
        if isinstance(key, int):
            n = int(key)
            if not self._signed:
                n = abs(n)
            s = str(n)
            if self._pad_to:
                s = s.zfill(self._pad_to)
            return s

        s = str(key)
        if self._lowercase:
            s = s.lower()
        return s

    def part_values(self, key: Hashable) -> PartValues:
        s = self._to_prefix_string(key)
        return tuple(s[i : i + 1] for i in range(self._depth))


class TimeIndex(PartitionIndex[Any]):
    def __init__(self, *, by_cols: Sequence[str], kind: str) -> None:
        super().__init__(by_cols=by_cols)
        self._kind = kind.replace('_', '-')

    def part_values(self, key: Any) -> PartValues:
        year = int(getattr(key, 'year'))
        month = getattr(key, 'month', None)
        day = getattr(key, 'day', None)

        if self._kind == 'year':
            return (year,)
        if self._kind == 'month':
            return (int(month),)
        if self._kind == 'day':
            return (int(day),)
        if self._kind == 'year-month':
            return (year, int(month))
        if self._kind == 'year-month-day':
            return (year, int(month), int(day))

        raise ValueError(f'Unsupported TimePartition kind: {self._kind!r}')


def infer_partition_index_for_table(table: Any, key_col: str) -> PartitionIndex[Any]:
    '''
    If the table is partitioned *by the same column* as the lookup key, infer
    an index so we can prune partitions.

    Otherwise return NoPartitionIndex().

    '''
    part = getattr(table, 'partitioning', None)
    if part is None:
        return NoPartitionIndex()

    # Only safe to map keys -> partitions if partitioner is derived from that key_col.
    if getattr(part, 'col', None) is None or part.col.name != key_col:
        return NoPartitionIndex()

    meta = part.meta
    by_cols = tuple(getattr(part, 'by_cols', ()))

    if isinstance(meta, MonoPartition):
        bucket_col = by_cols[0] if by_cols else 'bucket'
        return MonoBucketIndex(bucket_col=bucket_col, row_size=meta.row_size)

    if isinstance(meta, DictionaryPartition):
        return DictPrefixIndex(
            by_cols=by_cols,
            depth=meta.depth,
            pad_to=meta.pad_to,
            lowercase=meta.lowercase,
            signed=meta.signed,
        )

    if isinstance(meta, TimePartition):
        return TimeIndex(by_cols=by_cols, kind=meta.kind)

    return NoPartitionIndex()


# AccessPath (what to batch)


@dataclass(slots=True)
class AccessPath(Generic[TKey]):
    '''
    One batched lookup shape.

    `source` must have `.scan(use_cache=True)` returning LazyFrame (Table/Transform).
    After post/select, the result MUST contain the `key` column.

    '''
    name: str
    source: Any
    key: str

    partition: PartitionIndex[Any] | None = None
    post: PostFn | None = None
    select: Sequence[str | pl.Expr] | None = None

    def bind(self) -> None:
        if self.partition is None:
            self.partition = infer_partition_index_for_table(self.source, self.key)

    def build_query(self, part_values: PartValues, keys: set[TKey]) -> pl.LazyFrame:
        assert self.partition is not None, 'AccessPath.bind() must be called before use'

        lf = self.source.scan(use_cache=True)

        # prune partitions (if possible)
        lf = self.partition.apply_partition_filter(lf, part_values)

        # key filter (main lookup)
        lf = lf.filter(pl.col(self.key).is_in(list(keys)))

        # custom pipeline
        if self.post is not None:
            lf = self.post(lf)

        # projection
        if self.select is not None:
            lf = lf.select(list(self.select))

        return lf

    def empty_query(self) -> pl.LazyFrame:
        '''
        A 0-row query that still produces the correct output schema.
        (Important: limit FIRST so post ops don't scan.)

        '''
        lf = self.source.scan(use_cache=True).limit(0)
        if self.post is not None:
            lf = self.post(lf)
        if self.select is not None:
            lf = lf.select(list(self.select))
        return lf


# Task state & work state


@dataclass(slots=True)
class AccessTask(Generic[TKey]):
    id: int
    path: str
    keys: tuple[TKey, ...]

    # schema-correct empty result for this path
    empty_result: pl.DataFrame

    event: anyio.Event = field(default_factory=anyio.Event)
    result: pl.DataFrame | None = None
    error: Exception | None = None

    def unwrap(self) -> pl.DataFrame:
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError('Request completed without a result')
        return self.result


@dataclass(slots=True)
class AccessTaskState(Generic[TKey]):
    task: AccessTask[TKey]
    remaining: int
    frames: list[pl.DataFrame] = field(default_factory=list)
    done: bool = False


@dataclass(slots=True)
class PartitionWorkState(Generic[TKey]):
    '''
    Aggregated work for one (path, partition_values) group.

    keys_union: union of keys requested by tasks touching this partition.
    task_keys: per-task subset of keys for THIS partition.

    '''
    keys_union: set[TKey] = field(default_factory=set)
    task_keys: dict[int, set[TKey]] = field(default_factory=dict)


# Messages


@dataclass(slots=True)
class AccessRequestMessage(Generic[TKey]):
    task: AccessTask[TKey]


@dataclass(slots=True)
class CancelRequestMessage:
    task_id: int


AccessorMessage: TypeAlias = AccessRequestMessage[Any] | CancelRequestMessage


# DatasetAccessor


class DatasetAccessor(Generic[TKey]):
    '''
    Generic batching accessor with partition-aware grouping.

    '''

    def __init__(
        self,
        ctx: Context,
        paths: Sequence[AccessPath[Any]],
        *,
        executor: PolarsExecutor | None = None,
        batch_window_s: float = 0.005,
        max_buffer_size: int = 2048,
        log: Logger | None = None,
    ) -> None:
        self.ctx = ctx
        self._log = log
        self._executor = executor or PolarsExecutor(limit=1)
        self._batch_window_s = float(batch_window_s)

        self._task_id_gen = count()

        self._send, self._recv = anyio.create_memory_object_stream[AccessorMessage](
            max_buffer_size=max_buffer_size
        )

        self._paths: dict[str, AccessPath[Any]] = {p.name: p for p in paths}
        for p in self._paths.values():
            p.bind()

        self._tasks: dict[int, AccessTaskState[Any]] = {}

        self._work: dict[str, dict[PartValues, PartitionWorkState[Any]]] = {
            p.name: {} for p in self._paths.values()
        }

        # cached schema-correct empties per path
        self._empty_cache: dict[str, pl.DataFrame] = {}

    def _maybe_log(self, msg: str, *, level: str = 'info', **kwargs) -> None:
        if self._log is None:
            return
        getattr(self._log, level)(msg, **kwargs)

    async def _empty_for_path(self, path_name: str) -> pl.DataFrame:
        cached = self._empty_cache.get(path_name)
        if cached is not None:
            return cached

        path = self._paths[path_name]
        df = await self._executor.collect(path.empty_query())
        self._empty_cache[path_name] = df
        return df

    def _enqueue_task(self, task: AccessTask[Any]) -> None:
        path = self._paths.get(task.path)
        if path is None:
            task.error = KeyError(f'Unknown access path: {task.path!r}')
            task.event.set()
            return

        keys = tuple(task.keys)
        if not keys:
            task.result = task.empty_result
            task.event.set()
            return

        assert path.partition is not None

        key_groups = path.partition.groups_for_keys(keys)
        self._tasks[task.id] = AccessTaskState(task=task, remaining=len(key_groups))

        work_by_part = self._work[task.path]
        for pv, part_keys in key_groups.items():
            w = work_by_part.setdefault(pv, PartitionWorkState())
            w.keys_union.update(part_keys)
            w.task_keys.setdefault(task.id, set()).update(part_keys)

    def _cancel_task(self, task_id: int) -> None:
        state = self._tasks.pop(task_id, None)
        if state is None:
            return

        state.done = True
        state.task.error = anyio.get_cancelled_exc_class()()
        state.task.event.set()

        # best-effort removal from fanout sets
        for path_work in self._work.values():
            for w in path_work.values():
                w.task_keys.pop(task_id, None)

    async def _drain_for_batch(self) -> None:
        if self._batch_window_s <= 0:
            return

        with anyio.move_on_after(self._batch_window_s):
            while True:
                msg = await self._recv.receive()
                if isinstance(msg, CancelRequestMessage):
                    self._cancel_task(msg.task_id)
                else:
                    self._enqueue_task(msg.task)

    def _finish_ok(self, task_id: int) -> None:
        state = self._tasks.pop(task_id, None)
        if state is None or state.done:
            return

        state.done = True
        if not state.frames:
            state.task.result = state.task.empty_result
        elif len(state.frames) == 1:
            state.task.result = state.frames[0]
        else:
            state.task.result = pl.concat(state.frames, how='vertical')

        state.task.event.set()

    def _finish_err(self, task_id: int, err: Exception) -> None:
        state = self._tasks.pop(task_id, None)
        if state is None or state.done:
            return
        state.done = True
        state.task.error = err
        state.task.event.set()

    def _normalize_partition_by_keys(self, d: Mapping[Any, pl.DataFrame]) -> dict[Any, pl.DataFrame]:
        '''
        Polars partition_by(as_dict=True) key shape differs across versions.

        Some versions return scalar keys for single-column partitioning,
        others return 1-tuples like (123,). Normalize here.

        '''
        out: dict[Any, pl.DataFrame] = {}
        for k, v in d.items():
            if isinstance(k, tuple) and len(k) == 1:
                k = k[0]
            elif isinstance(k, list) and len(k) == 1:
                k = k[0]
            out[k] = v
        return out

    def _group_rows_by_key(self, df: pl.DataFrame, key_col: str) -> Mapping[Any, pl.DataFrame]:
        '''
        key -> DataFrame mapping used for routing.

        IMPORTANT FIX: normalize dict keys from partition_by() so lookups
        with python ints/strings work across Polars versions.

        '''
        if df.height == 0:
            return {}

        if key_col not in df.columns:
            raise RuntimeError(
                f'Accessor routing failed: result missing key column {key_col!r}. columns={df.columns}'
            )

        part_by = getattr(df, 'partition_by', None)
        if callable(part_by):
            try:
                d = df.partition_by(key_col, as_dict=True, maintain_order=True)  # type: ignore[misc]
            except TypeError:
                d = df.partition_by(key_col, as_dict=True)  # type: ignore[misc]
            return self._normalize_partition_by_keys(d)

        # fallback: python index map
        keys = df[key_col].to_list()
        idx_map: dict[Any, list[int]] = {}
        for i, k in enumerate(keys):
            idx_map.setdefault(k, []).append(i)

        take = getattr(df, 'take', None)
        if callable(take):
            return {k: df.take(idxs) for k, idxs in idx_map.items()}  # type: ignore[misc]

        # last-resort fallback
        return {k: df.filter(pl.col(key_col) == k) for k in idx_map}

    async def _flush_path(self, path_name: str) -> None:
        path = self._paths[path_name]
        work = self._work[path_name]
        if not work:
            return

        # swap out so new work can accumulate immediately
        self._work[path_name] = {}

        items = [(pv, w) for (pv, w) in work.items() if w.task_keys]
        if not items:
            return

        lazy_frames = tuple(path.build_query(pv, w.keys_union) for pv, w in items)

        # collect batched; fall back to isolate errors
        try:
            data_frames = await self._executor.collect_all(lazy_frames)
            errors: list[Exception | None] = [None] * len(items)
        except Exception:
            data_frames = []
            errors = []
            for lf in lazy_frames:
                try:
                    data_frames.append(await self._executor.collect(lf))
                    errors.append(None)
                except Exception as e:
                    data_frames.append(path.empty_query().collect())  # schema-correct empty
                    errors.append(e)

        for (pv, w), df, err in zip(items, data_frames, errors, strict=True):
            if err is not None:
                for task_id in list(w.task_keys.keys()):
                    self._finish_err(task_id, err)
                continue

            # route df -> tasks
            grouped = self._group_rows_by_key(df, path.key)

            for task_id, task_keys in list(w.task_keys.items()):
                state = self._tasks.get(task_id)
                if state is None or state.done:
                    continue

                parts = [grouped.get(k) for k in task_keys]
                parts = [p for p in parts if p is not None and p.height > 0]

                if parts:
                    state.frames.append(parts[0] if len(parts) == 1 else pl.concat(parts, how='vertical'))

                state.remaining -= 1
                if state.remaining <= 0:
                    self._finish_ok(task_id)

    async def _flush(self) -> None:
        for name in self._paths.keys():
            await self._flush_path(name)

    async def _collector_task(self) -> None:
        async with self._recv:
            while True:
                try:
                    msg = await self._recv.receive()
                except anyio.EndOfStream:
                    break

                if isinstance(msg, CancelRequestMessage):
                    self._cancel_task(msg.task_id)
                else:
                    self._enqueue_task(msg.task)

                await self._drain_for_batch()
                await self._flush()

        for task_id in list(self._tasks.keys()):
            self._finish_err(task_id, RuntimeError('Accessor shutting down'))

    def _try_cancel(self, task_id: int) -> None:
        try:
            self._send.send_nowait(CancelRequestMessage(task_id=task_id))
        except (anyio.WouldBlock, anyio.ClosedResourceError):
            return

    async def request(self, path: str, keys: TKey | Sequence[TKey]) -> pl.DataFrame:
        if isinstance(keys, (list, tuple)):
            key_tuple = tuple(keys)
        else:
            key_tuple = (keys,)

        if path not in self._paths:
            raise KeyError(f'Unknown access path: {path!r}')

        empty_result = await self._empty_for_path(path)

        task = AccessTask(
            id=next(self._task_id_gen),
            path=path,
            keys=key_tuple,
            empty_result=empty_result,
        )

        await self._send.send(AccessRequestMessage(task=task))

        try:
            await task.event.wait()
        except BaseException:
            self._try_cancel(task.id)
            raise

        return task.unwrap()

    async def aclose(self) -> None:
        await self._send.aclose()


# Async context helper


from contextlib import asynccontextmanager
from typing import AsyncGenerator


@asynccontextmanager
async def open_accessor(
    ctx: Context,
    paths: Sequence[AccessPath[Any]],
    *,
    executor: PolarsExecutor | None = None,
    batch_window_s: float = 0.005,
    max_buffer_size: int = 2048,
    log: Logger | None = None,
) -> AsyncGenerator[DatasetAccessor[Any], None]:
    accessor = DatasetAccessor(
        ctx,
        paths,
        executor=executor,
        batch_window_s=batch_window_s,
        max_buffer_size=max_buffer_size,
        log=log,
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(accessor._collector_task)
        try:
            yield accessor
        finally:
            await accessor.aclose()
