from typing import Sequence

import anyio
import polars as pl


class PolarsExecutor:
    def __init__(
        self,
        limit: int = 1
    ) -> None:
        self._limit = anyio.CapacityLimiter(limit)


    async def collect(
        self,
        frame: pl.LazyFrame
    ) -> pl.DataFrame:
        async with self._limit:
            return await frame.collect_async()

    async def collect_all(
        self,
        frames: Sequence[pl.LazyFrame]
    ) -> list[pl.DataFrame]:
        async with self._limit:
            return await pl.collect_all_async(frames)
