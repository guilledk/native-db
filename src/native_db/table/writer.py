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
