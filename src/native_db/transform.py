import msgspec

import polars as pl

from native_db.table import Table


class Transform(msgspec.Struct, frozen=True):
    '''
    Declare a dataframe transform using a pl.LazyFrame, and a cache path and
    provide a cached scan_parquet api to the materialized dataframe.

    Usefull to pre-cache expensive queries that are just views into our static
    data.

    '''

    query: pl.LazyFrame
    meta: Table

    @property
    def is_cached(self) -> bool:
        return self.meta.exists

    @property
    def disk_size(self) -> int:
        return self.meta.disk_size

    def scan(self) -> pl.LazyFrame:
        '''
        Materialize transform to meta.local_path if not present already, then
        return a `pl.LazyFrame` to it.

        '''
        if not self.is_cached:
            self.query.sink_parquet(
                self.meta.local_path,
                compression=self.meta.compression,
                row_group_size=self.meta.schema.row_group_size,
                mkdir=True,
            )

        return pl.scan_parquet(self.meta.local_path)
