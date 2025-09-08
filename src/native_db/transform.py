from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable
import shutil

from native_db.lowlevel.diskops import FrameFormats, sink_frame, scan_frame
import polars as pl

from native_db._utils import path_size

if TYPE_CHECKING:
    from native_db._ctx import Context


@runtime_checkable
class LambdaQuery(Protocol):
    def __call__(self, ctx: 'Context') -> pl.LazyFrame:
        ...


class Transform:
    '''
    Declare a dataframe transform using a pl.LazyFrame, and a cache path and
    provide a cached scan_parquet api to the materialized dataframe.

    Usefull to pre-cache expensive queries that are just views into our static
    data.

    '''
    def __init__(
        self,
        name: str,
        query: pl.LazyFrame | LambdaQuery,
        cache_path: Path,
        cache_format: FrameFormats = 'parquet',
        sink_args: dict[str, Any] = {},
        *,
        partition_by: list[str] | None = None,
        include_key: bool = False,
        per_partition_sort_by: list[str] | None = None,
        # Optional "prepare" to add derived partition columns (e.g., bucket)
        prepare: Callable | None = None,
    ) -> None:
        self.name = name
        self.query = query
        self.cache_path = cache_path
        self.cache_format: FrameFormats = cache_format
        self.sink_args = sink_args
        self.partition_by = partition_by
        self.include_key = include_key
        self.per_partition_sort_by = per_partition_sort_by or []
        self.prepare = prepare

        self._frame: pl.LazyFrame | None = None

    @property
    def is_cached(self) -> bool:
        return self.cache_path.exists()

    @property
    def disk_size(self) -> int:
        return path_size(self.cache_path)

    def clear_cache(self) -> None:
        if self.cache_path.is_dir():
            return shutil.rmtree(self.cache_path)

        else:
            self.cache_path.unlink(missing_ok=True)

    def scan(self, ctx: 'Context | None' = None, use_cache: bool = True) -> pl.LazyFrame:
        '''
        Materialize transform to meta.local_path if not present already, then
        return a `pl.LazyFrame` to it.

        '''
        if use_cache and self._frame is not None:
            return self._frame

        if not self.is_cached:
            query = self.query
            if isinstance(query, LambdaQuery):
                if not ctx:
                    raise FileNotFoundError("Transform.scan requires ctx for LambdaQuery")
                lf = query(ctx)
            else:
                lf = query

            # Optionally add derived partition keys
            if self.prepare is not None:
                lf = self.prepare(lf)

            # Stream to disk (file or partitioned directory)
            if self.partition_by:
                tmp_dir = self.cache_path.with_name(self.cache_path.name + ".tmp")
                if tmp_dir.exists():
                    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
                res = sink_frame(
                    lf,
                    pl.PartitionByKey(
                        tmp_dir,
                        by=self.partition_by,
                        include_key=self.include_key,
                        per_partition_sort_by=[pl.col(c) for c in self.per_partition_sort_by],
                    ),
                    format='parquet',
                    **self.sink_args,
                )
                # execute streaming sink
                _ = res.collect()
                # atomic-ish replace directory
                import shutil, os
                if self.cache_path.exists():
                    shutil.rmtree(self.cache_path, ignore_errors=True)
                os.replace(tmp_dir, self.cache_path)
            else:
                # single-file cache (existing behavior)
                res = sink_frame(lf, self.cache_path, format=self.cache_format, **self.sink_args)
                _ = res.collect()

        # cache & return a lazy scan into the cache
        if self.partition_by:
            # directory scan with hive partition parsing enables pruning by path
            frame = scan_frame(
                self.cache_path,
                format='parquet',
                hive_partitioning=True,
            )
        else:
            frame = scan_frame(self.cache_path)

        self._frame = frame

        return frame
