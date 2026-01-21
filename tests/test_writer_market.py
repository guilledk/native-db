import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import polars as pl
import pytest

from native_db._ctx import Context, ContextBuilder, open_ctx_writer
from native_db._testing import MarketContext, market_ctx, market_frame_stream
from native_db.table import Table
from native_db.table._layout import TimePartKind, TimePartition
from native_db.table.writer import TableWriterOptions


def _mk_ctx(
    tmp_path: Path, kind: TimePartKind,
    rows_per_file: int,
    commit_threshold: int
) -> MarketContext:
    # copy the catalog entry, but point to a fresh datadir and desired partition kind
    ctx = market_ctx(tmp_path)
    t = ctx.market.copy(
        partitioning=TimePartition(on_column='time', kind=kind),
        writer_opts=TableWriterOptions(
            rows_per_file=rows_per_file, commit_threshold=commit_threshold
        )
    )
    # ensure a clean slate
    if t.local_path.exists():
        for p in sorted(t.local_path.rglob('*'), reverse=True):
            if p.is_file():
                p.unlink()
            else:
                p.rmdir()
        t.local_path.rmdir()
    return Context(
        datadir=tmp_path,
        tables=(t, ),
        transforms=tuple()
    )  # type: ignore


async def _write_stream(
    ctx: MarketContext,
    row_stream: Generator[tuple[list, int], None, None],
) -> None:
    builder = ContextBuilder(ctx)
    for rows, i in row_stream:
        builder.extend('market', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

    staging_dir = ctx.market.local_path / '.staging'
    staged_ipcs = list(staging_dir.glob('*.ipc'))
    # drain should leave no frames in staging area
    assert not staged_ipcs


def _list_partitions(root: Path) -> set[str]:
    return {
        p.name for p in root.iterdir() if p.is_dir() and p.name != '.staging'
    }


def _expected_partitions(
    kind: str, start: datetime, periods: int, step: timedelta
) -> set[str]:
    vals = set()
    t = start
    for _ in range(periods):
        if kind == 'year':
            vals.add(f'year={t.year}')
        elif kind == 'month':
            vals.add(f'month={t.month}')  # Polars writes raw integer value
        elif kind == 'day':
            vals.add(f'day={t.day}')
        else:
            raise AssertionError(kind)
        t = t + step
    return vals


def _assert_partition_values(table: Table, kind: str) -> None:
    # For every partition bucket, load its parquet files and assert that the time-derived value matches the bucket key
    for bucket in table.local_path.iterdir():
        if not bucket.is_dir() or bucket.name == '.staging':
            continue
        name = bucket.name  # e.g. "year=2024" or "month=7"
        key, val_str = name.split('=')
        val = int(val_str)
        lf = pl.scan_parquet(str(bucket / table.file_pattern))
        derived = getattr(pl.col('time').dt, key)()
        # each file is already sorted within partition; just verify all derived values equal the partition key
        ok = lf.select((derived == val).all()).collect().item()
        assert ok, f'Partition {name} has rows with mismatching {key} values'


@pytest.mark.parametrize(
    'kind,start,periods,step',
    [
        (
            'year',
            datetime(2023, 12, 20, tzinfo=timezone.utc),
            401,
            timedelta(days=1),
        ),  # spans into 2024
        (
            'month',
            datetime(2024, 1, 28, tzinfo=timezone.utc),
            10,
            timedelta(days=1),
        ),  # spans months
        (
            'day',
            datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
            20,
            timedelta(hours=6),
        ),  # spans multiple days
    ],
)
async def test_time_partitioning_market_table(
    anyio_backend,
    tmp_path: Path,
    kind: TimePartKind,
    start: datetime,
    periods: int,
    step: timedelta,
) -> None:
    # choose sizes so total rows is a multiple of commit_threshold -> staging empties completely
    rows_per_file = 50
    commit_threshold = 200
    ctx = _mk_ctx(tmp_path / '.test-db', kind, rows_per_file, commit_threshold)
    total = periods
    # if periods not divisible, bump it up so test is robust
    if total % commit_threshold != 0:
        total = int(math.ceil(total / commit_threshold) * commit_threshold)

    frames = market_frame_stream(
        start=start,
        periods=total,
        step=step,
        batch_size=37,
        seed=123,
        start_price=123.45,
        vol=0.9,
    )

    await _write_stream(
        ctx,
        frames,
    )

    # expected set of partition directories present
    exp = _expected_partitions(kind, start, total, step)
    got = _list_partitions(ctx.market.local_path)
    assert exp == got, f'expected partitions {exp}, got {got}'

    # per-partition correctness (rows obey the time bucket)
    _assert_partition_values(ctx.market, kind)
