import fnmatch
import os
import shutil
from pathlib import Path
from logging import Logger, getLogger
from typing import TYPE_CHECKING

import anyio

from native_db.lowlevel import PolarsExecutor
from native_db.lowlevel.diskops import (
    FrameFormats,
    concat_or_split_frame,
    find_last_part,
)
from native_db.structs import FrozenStruct

if TYPE_CHECKING:
    from native_db import Table

log = getLogger(__name__)


class StagedPart(FrozenStruct, frozen=True):
    """
    A single partitioned file produced by a builder, ready to be integrated.
    - path: on-disk file path inside a builder's commit directory
    - rel_parent: partition subdir relative to the commit root
                  (e.g., 'bucket=1234' or 'year=2024/month=09')
    - commit_root: the commit directory that holds this part
    - format/number_of_rows: informational
    """
    path: Path
    rel_parent: Path
    commit_root: Path
    format: FrameFormats
    number_of_rows: int = 0


class TableWriterOptions(FrozenStruct, frozen=True):
    # NOTE: commit_threshold/stage_threshold are no-ops in the new model.
    # We integrate parts as they arrive.
    commit_threshold: int = 500_000
    stage_threshold: int = 500_000
    # number of rows per final file inside each partition
    rows_per_file: int = 500_000
    # kept for backward compatibility; unused
    start_frame_index: int = 0


class TableWriter:
    """
    New writer: integrate partitioned parts into the live dataset.

    Builders now materialize pre-partitioned Parquet (or IPC/CSV) parts under:
        <table>/.staging/pid-<PID>/commit-<N>/<hive-keys...>/<part-file>

    The writer only needs to *merge* those files into the table path, rolling
    according to rows_per_file. No global frame indexing or re-ordering.
    """

    def __init__(
        self,
        table: "Table",
        options: TableWriterOptions | None = None,
        *,
        log: Logger = log,
        executor: PolarsExecutor | None = None,
    ):
        self.log = log
        self.options = options or TableWriterOptions()
        self._table = table
        self._executor = executor or PolarsExecutor()
        self._staging_path = table.local_path / ".staging"
        # commit_root -> remaining files (discovered lazily)
        self._commit_refcnt: dict[Path, int] = {}
        self._lock = anyio.Lock()

    async def _frame_into_bucket(self, src: Path, bucket_path: Path) -> None:
        bucket_path.mkdir(parents=True, exist_ok=True)
        last, last_idx = find_last_part(bucket_path, self._table.file_pattern)
        if not last:
            dst = bucket_path / self._table.part_file(0)
            src.replace(dst)
            return

        await concat_or_split_frame(
            src,
            target=last,
            split=bucket_path / self._table.part_file(last_idx + 1),
            max_rows=self.options.rows_per_file,
            executor=self._executor,
        )

    def _discover_commit_count(self, commit_root: Path) -> int:
        count = 0
        for dirpath, _dirs, files in os.walk(commit_root):
            for name in files:
                if fnmatch.fnmatch(name, self._table.file_pattern):
                    count += 1
        return count

    async def stage_direct(self, part: StagedPart) -> None:
        """
        Integrate one partitioned part into the live dataset.
        """
        async with self._lock:
            # Establish expected count for this commit_root on first sight.
            if part.commit_root not in self._commit_refcnt:
                self._commit_refcnt[part.commit_root] = self._discover_commit_count(
                    part.commit_root
                )

        # Move/concat the part into its final bucket.
        bucket = self._table.local_path / part.rel_parent
        await self._frame_into_bucket(part.path, bucket)

        # Decrement and possibly clean commit_root.
        async with self._lock:
            left = self._commit_refcnt.get(part.commit_root, 0) - 1
            if left <= 0:
                self._commit_refcnt.pop(part.commit_root, None)
                shutil.rmtree(part.commit_root, ignore_errors=True)
            else:
                self._commit_refcnt[part.commit_root] = left

    async def drain(self) -> None:
        # Best-effort cleanup of the whole staging area on close.
        shutil.rmtree(self._staging_path, ignore_errors=True)

# import os
# import shutil
# import fnmatch
# 
# from itertools import count
# from collections import deque
# 
# from pathlib import Path
# from logging import Logger, getLogger
# from typing import TYPE_CHECKING, Deque, Self
# 
# import anyio
# import polars as pl
# 
# 
# from native_db.lowlevel import PolarsExecutor
# from native_db.lowlevel.diskops import FrameFormats, concat_or_split_frame, find_last_part, scan_frame, sink_frame
# from native_db.structs import FrozenStruct
# 
# if TYPE_CHECKING:
#     from native_db import Table
# 
# 
# log = getLogger(__name__)
# 
# 
# class StagedFrame(FrozenStruct, frozen=True):
#     index: int
#     number_of_rows: int
#     path: Path
#     format: FrameFormats
# 
#     @classmethod
#     def from_push(
#         cls,
#         path: str | Path,
#         index: int | None = None,
#         number_of_rows: int | None = None,
#         format: FrameFormats | None = None
#     ) -> Self:
# 
#         if isinstance(path, str):
#             path = Path(path)
# 
#         # if frame metadata is not passed directly, figure out from file name
#         # format must be: f'frame-{frame_index}.{number_of_rows}-rows.{format}'
#         if (
#             index is None
#             or number_of_rows is None
#             or format is None
#         ):
#             idx_part, rows_part, suffix = path.name.split('.')
#             _, idx = idx_part.split('-')
#             rows, _ = rows_part.split('-')
# 
#             if index is None:
#                 index = int(idx)
# 
#             if not number_of_rows:
#                 number_of_rows = int(rows)
# 
#             if not format:
#                 assert suffix in ('ipc', 'parquet', 'csv')
#                 format = suffix
# 
#         return cls(
#             index=index,
#             number_of_rows=number_of_rows,
#             path=path,
#             format=format
#         )
# 
#     @property
#     def framename(self) -> str:
#         return f'frame-{self.index:05d}.{self.number_of_rows}-rows.{self.format}'
# 
#     def scan(self) -> pl.LazyFrame:
#         return scan_frame(self.path, format=self.format)
# 
# 
# class TableWriterOptions(FrozenStruct, frozen=True):
#     # number of rows in staging before triggering commit
#     commit_threshold: int = 500_000
#     stage_threshold: int = 500_000
#     # number of rows per file, post partitioning
#     rows_per_file: int = 500_000
#     # set frame index sequence start
#     start_frame_index: int = 0
# 
# 
# class TableWriter:
#     '''
#     # Overview
# 
#     Helper for stream-writing tables larger than memory while respecting the
#     semantics emposed by the hint system (check `native_db.dtypes` module
#     docstring for information on hint semantics).
# 
#     Only a single writer is allowed per table directory.
# 
#     '''
# 
#     def __init__(
#         self,
#         table: 'Table',
#         options: TableWriterOptions | None = None,
#         *,
#         log: Logger = log,
#         executor: PolarsExecutor | None = None
#     ):
#         self.log = log
#         self.options = options or TableWriterOptions()
# 
#         self._table = table
#         self._part = table.partitioning
# 
#         self._executor = executor or PolarsExecutor()
# 
#         # when push is gonna be called with in order frames, an autogenerated
#         # frame index is used provided by itertools.count
#         self._frame_index: count[int] = count(self.options.start_frame_index)
# 
#         self._staging_path = table.local_path / '.staging'
#         self._staging: dict[int, StagedFrame] = {}
#         self._staged_rows: int = 0
#         self._commit_index: count[int] = count(0)
# 
#         self._next_expected_index: int = self.options.start_frame_index
#         self._in_order_staging: Deque[StagedFrame] = deque()
#         self._commit_lock = anyio.Lock()
# 
#     async def _frame_into_bucket(self, src: Path, bucket_path: Path) -> None:
#         bucket_path.mkdir(parents=True, exist_ok=True)
#         last, last_idx = find_last_part(bucket_path, self._table.file_pattern)
#         if not last:
#             dst = bucket_path / self._table.part_file(0)
#             src.replace(dst)
#             return
# 
#         await concat_or_split_frame(
#             src,
#             target=last,
#             split=bucket_path / self._table.part_file(last_idx + 1),
#             max_rows=self.options.rows_per_file,
#             executor=self._executor
#         )
# 
#     async def _commit(
#         self,
#         frames: list[StagedFrame],
#         commit_root: Path
#     ) -> None:
#         if not frames:
#             return
# 
#         # Build a single lazy concat of all staged IPC files (streaming sink).
#         staged_frame = pl.concat([sf.scan() for sf in frames], rechunk=False)
# 
#         commit_root.mkdir(parents=True, exist_ok=True)
# 
#         try:
#             if self._part:
#                 staged_frame = self._part.prepare(staged_frame)
# 
#                 # Preserve any hinted sorts (or at least sort by partition key)
#                 sort_cols = [
#                     pl.col(col.name)
#                     for col in self._table.schema.columns
#                     if col.hints.sort in ('asc', 'desc')
#                 ]
#                 per_part_sort = sort_cols or []
# 
# 
#                 await self._executor.collect(
#                     sink_frame(
#                         staged_frame,
#                         pl.PartitionParted(
#                             commit_root,
#                             by=self._part.by_cols,
#                             include_key=False,
#                             per_partition_sort_by=per_part_sort,
#                         ),
#                         **self._table.sink_args()
#                     )
#                 )
#                 # Integrate each produced part into its final bucket
#                 for dirpath, _dirs, files in os.walk(commit_root):
#                     for name in files:
#                         if name.startswith('frame-'):
#                             continue
# 
#                         if fnmatch.fnmatch(name, self._table.file_pattern):
#                             src_part = Path(dirpath) / name
#                             rel_parent = Path(dirpath).relative_to(commit_root)
#                             await self._frame_into_bucket(src_part, self._table.local_path / rel_parent)
#             else:
#                 tmp_location = commit_root / f'commit.{self._table.format}'
#                 await self._executor.collect(
#                     sink_frame(
#                         staged_frame,
#                         tmp_location,
#                         **self._table.sink_args()
#                     )
#                 )
#                 await self._frame_into_bucket(tmp_location, self._table.local_path)
#         finally:
#             shutil.rmtree(commit_root, ignore_errors=True)
# 
#     async def _commit_in_chunks(self, final: bool = False) -> None:
#         while self._in_order_staging and (final or self._staged_rows >= self.options.commit_threshold):
#             commit_id = next(self._commit_index)
#             sub_stage = self._staging_path / f'.commit-{commit_id}'
#             sub_stage.mkdir(parents=True, exist_ok=True)
# 
#             # pack up to threshold rows (or all, on final)
#             commit_frames = []
#             commit_rows = 0
#             while self._in_order_staging and (final or commit_rows < self.options.commit_threshold):
#                 sf = self._in_order_staging.popleft()
#                 sub_path = sub_stage / sf.framename
#                 os.rename(sf.path, sub_path)
#                 commit_frames.append(StagedFrame(index=sf.index, path=sub_path, number_of_rows=sf.number_of_rows, format=sf.format))
#                 commit_rows += sf.number_of_rows
# 
#             self._staged_rows -= commit_rows
#             if self._staged_rows < 0: self._staged_rows = 0
# 
#             async with self._commit_lock:
#                 await self._commit(commit_frames, sub_stage)
# 
#     async def stage_direct(self, frame: StagedFrame) -> None:
#         self._staging[frame.index] = frame
#         while self._next_expected_index in self._staging:
#             next_frame = self._staging[self._next_expected_index]
#             self._in_order_staging.append(next_frame)
#             self._staged_rows += next_frame.number_of_rows
#             self._next_expected_index += 1
#             del self._staging[next_frame.index]
# 
#         # normal threshold-based commits (unchanged)
#         if self._staged_rows >= self.options.commit_threshold:
#             await self._commit_in_chunks()
# 
#     async def drain(self) -> None:
#         for next_frame in sorted(self._staging.values(), key=lambda f: f.index):
#             self._in_order_staging.append(next_frame)
# 
#         await self._commit_in_chunks(final=True)
# 
#         # Only remove our commit scratch dirs; leave producer staging alone.
#         staging = self._table.local_path / '.staging'
#         try:
#             for p in staging.glob('.commit-*'):
#                 shutil.rmtree(p, ignore_errors=True)
#             # Best-effort: if staging is now empty, remove it; otherwise keep it.
#             next(staging.iterdir())
#         except StopIteration:
#             shutil.rmtree(staging, ignore_errors=True)
#         except FileNotFoundError:
#             # already gone; nothing to do
#             pass
# 
#     async def stage(
#         self,
#         path: str | Path,
#         *,
#         index: int | None = None,
#         number_of_rows: int = 0,
#         format: FrameFormats | None = None
#     ) -> None:
#         await self.stage_direct(StagedFrame.from_push(
#             path,
#             index=index,
#             number_of_rows=number_of_rows,
#             format=format
#         ))
# 
#     async def stage_frame(
#         self,
#         frame: pl.DataFrame,
#         index: int,
#         *,
#         format: FrameFormats = 'ipc'
#     ) -> None:
#         path = self._table.local_path / '.staging' / f'frame-{index:05d}.{frame.height}.{format}'
# 
#         await self._executor.collect(
#             sink_frame(
#                 frame.lazy(),
#                 path,
#                 format=format
#             )
#         )
# 
#         await self.stage_direct(StagedFrame.from_push(
#             path,
#             index=index,
#             number_of_rows=frame.height,
#             format=format
#         ))
