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


@runtime_checkable
class LambdaBoolQuery(Protocol):
    def __call__(self, ctx: 'Context') -> bool:
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
        primary_key: str | None = None,
        partition_by: list[str] | None = None,
        include_key: bool = False,
        per_partition_sort_by: list[str] | None = None,
        # Optional "prepare" to add derived partition columns (e.g., bucket)
        prepare: Callable | None = None,
        is_cached: LambdaBoolQuery | None = None,
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
        self._is_cached = is_cached
        self.primary_key = primary_key

        self._frame: pl.LazyFrame | None = None

    def is_cached(self, ctx: 'Context | None' = None) -> bool:
        if self._is_cached:
            return self._is_cached(ctx)

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

        is_cached = self.is_cached(ctx)

        if not is_cached:
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

                if self.per_partition_sort_by:
                    sort_by = list(self.partition_by)
                    for c in self.per_partition_sort_by:
                        if c not in sort_by:
                            sort_by.append(c)
                    lf = lf.sort(by=sort_by)

                res = sink_frame(
                    lf,
                    pl.PartitionBy(
                        tmp_dir,
                        include_key=self.include_key,
                        key=self.partition_by,
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
                tmp_cache = self.cache_path.with_name(self.cache_path.name + ".tmp")
                if tmp_cache.exists():
                    import shutil; shutil.rmtree(tmp_cache, ignore_errors=True)
                try:
                    # check if cache exists in disk
                    cached_lf = scan_frame(self.cache_path)
                    cached_lf.collect()

                    idx = self.primary_key
                    new_rows_lf = (
                        lf
                        .join(
                            cached_lf.select(idx),
                            on=idx,
                            how='left',
                        )
                    )
                    lf = (
                        pl.concat(
                            (cached_lf, new_rows_lf),
                            how='vertical',
                            rechunk=False,
                        )
                        .unique(idx, maintain_order=True, keep='last')
                    )

                except FileNotFoundError:
                    # cache not created yet
                    pass

                finally:
                    res = sink_frame(lf, tmp_cache, format=self.cache_format, **self.sink_args)
                    _ = res.collect()
                    tmp_cache.rename(self.cache_path)


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
