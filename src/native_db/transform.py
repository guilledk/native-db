from pathlib import Path
from typing import Any

import polars as pl

from native_db._utils import path_size


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
        query: pl.LazyFrame,
        cache_path: Path,
        sink_args: dict[str, Any] = {}
    ) -> None:
        self.name = name
        self.query = query
        self.cache_path = cache_path
        self.sink_args = sink_args

        self._frame: pl.LazyFrame | None = None

    @property
    def is_cached(self) -> bool:
        return self.cache_path.exists()

    @property
    def disk_size(self) -> int:
        return path_size(self.cache_path)

    def scan(self, use_cache: bool = True) -> pl.LazyFrame:
        '''
        Materialize transform to meta.local_path if not present already, then
        return a `pl.LazyFrame` to it.

        '''
        if use_cache and self._frame is not None:
            return self._frame

        if not self.is_cached:
            self.query.sink_parquet(
                self.cache_path,
                mkdir=True,
                **self.sink_args
            )

        frame = pl.scan_parquet(self.cache_path)
        if use_cache:
            self._frame = frame

        return frame
