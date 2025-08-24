from datetime import date, datetime, timedelta
from pathlib import Path
import random
from typing import Generator, Literal

import polars as pl

from native_db._utils import epoch
from native_db.dtypes import Keyword, Mono, TypeHints
from native_db.table import Table
from native_db.table._layout import KeywordPartitionMeta, MonoPartitionMeta, TimePartitionMeta


block_table = Table(
    'block',
    'static/blocks',
    (
        ('number', Mono(size=4), TypeHints(sort='asc')),
        ('timestamp', pl.Datetime(time_unit='us', time_zone='UTC')),
        ('hash', Keyword, TypeHints(avg_str_size=64)),
    ),
    datadir=Path(__file__).parent.parent.parent / 'tests/.test-db',
    partitioning=MonoPartitionMeta(on_column='number')
)


block_time_step = timedelta(seconds=0.5)


def block_stream(
    start_number: int = 0,
    end_number: int = 10_000,
    start_date: datetime = epoch,
    time_step: timedelta = block_time_step,
) -> Generator[tuple[int, datetime, str], None, None]:
    total_blocks = end_number - start_number
    if total_blocks <= 0:
        raise ValueError('start_number must be < than end_number')

    h: str = '00' * 32

    for i in range(total_blocks):
        hex_i = hex(i)[2:]  # hex i repr without 0x
        yield (
            start_number + i,
            start_date + (time_step * i),
            hex_i + h[len(hex_i) :],
        )


def block_frame_stream(
    *,
    start_number: int = 0,
    end_number: int = 100_000,
    start_date: datetime = epoch,
    time_step: timedelta = block_time_step,
    batch_size: int,
    order: Literal['ordered', 'out_of_order'] = 'ordered',
    seed: int | None = None,
    # limit how "wild" out-of-order can be (windowed shuffle)
    max_reorder_window: int | None = None,
) -> Generator[tuple[pl.DataFrame, int], None, None]:
    '''
    Yield batches as (DataFrame, frame_index).
    - order='ordered': emit in natural increasing frame order.
    - order='out_of_order': emit in shuffled order (optionally bounded).

    '''
    schema = block_table.schema.as_polars()
    # 1. Build all batches
    rows_iter = block_stream(start_number, end_number, start_date, time_step)
    batches: list[tuple[pl.DataFrame, int]] = []
    idx = 0
    buf = []
    for r in rows_iter:
        buf.append(r)
        if len(buf) == batch_size:
            df = pl.DataFrame(buf, orient='row', schema=schema)
            batches.append((df, idx))
            idx += 1
            buf = []
    if buf:
        df = pl.DataFrame(buf, orient='row', schema=schema)
        batches.append((df, idx))

    # 2. Decide emission order
    if order == 'ordered' or len(batches) <= 1:
        emit_order = list(range(len(batches)))
    else:
        rnd = random.Random(seed)
        if max_reorder_window and max_reorder_window > 1:
            emit_order = []
            for start in range(0, len(batches), max_reorder_window):
                window = list(
                    range(start, min(start + max_reorder_window, len(batches)))
                )
                rnd.shuffle(window)
                emit_order.extend(window)
        else:
            emit_order = list(range(len(batches)))
            rnd.shuffle(emit_order)

    # 3. Yield
    for i in emit_order:
        yield batches[i]


pubmed_table = Table(
    'pubmed',
    'static/pubmed',
    (
        ('pmid', Mono(size=8), TypeHints(sort='asc')),
        ('pub_date', pl.Date),
        ('title', pl.String, TypeHints(avg_str_size=128)),
    ),
    datadir=Path(__file__).parent.parent / 'tests/.test-db',
    partitioning=MonoPartitionMeta(on_column='pmid')
)


def pubmed_frame_stream(
    *,
    # list of (start, end) inclusive ranges, emitted in order; gaps may exist between ranges
    ranges: list[tuple[int, int]],
    batch_size: int,
) -> Generator[tuple[pl.DataFrame, int], None, None]:
    '''
    Yield (DataFrame, frame_index) batches representing a set of ordered pmid ranges,
    preserving local order and allowing gaps across ranges.

    - The frame_index is strictly increasing (0..N-1).
    - Titles are synthetic; pub_date increments daily for determinism.

    '''
    schema = pubmed_table.schema.as_polars()
    rows: list[tuple[int, date, str]] = []

    # deterministic dates/titles
    cur_date = epoch.date()
    for lo, hi in ranges:
        for pmid in range(lo, hi + 1):
            rows.append((pmid, cur_date, f'title-{pmid}'))
            cur_date = cur_date + timedelta(days=1)

    batches: list[tuple[pl.DataFrame, int]] = []
    idx = 0
    buf: list[tuple[int, date, str]] = []
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            df = pl.DataFrame(buf, orient='row', schema=schema)
            batches.append((df, idx))
            idx += 1
            buf = []
    if buf:
        df = pl.DataFrame(buf, orient='row', schema=schema)
        batches.append((df, idx))

    for b in batches:
        yield b


pubmed_author_table = Table(
    'pubmed_author',
    'static/pubmed_author',
    (
        ('name', Keyword),
        ('pmid', Mono(size=8), TypeHints(sort='asc')),
    ),
    datadir=Path(__file__).parent.parent / 'tests/.test-db',
    partitioning=KeywordPartitionMeta(on_column='name', char_depth=1),
)


def pubmed_author_random_frame_stream(
    *,
    total_rows: int,
    batch_size: int,
    seed: int = 123,
    authors: list[str] | None = None,
    pmid_start: int = 1,
) -> Generator[tuple[pl.DataFrame, int], None, None]:
    """
    Yield (DataFrame, frame_index) batches matching pubmed_author_table's schema.

    - author_name: sampled from `authors` (or a default list).
    - pmid: increasing Mono32-compatible ints starting at pmid_start.
    """
    rng = random.Random(seed)
    if not authors:
        # keep ASCII-ish to avoid unicode slicing surprises in tests
        authors = ['Alice', 'Bob', 'Charlie', 'Dan', 'Eve', 'Mallory', 'Oscar']

    schema = pubmed_author_table.schema.as_polars()

    rows: list[tuple[str, int]] = []
    pmid = pmid_start
    for _ in range(total_rows):
        rows.append((rng.choice(authors), pmid))
        pmid += 1

    # batch
    idx = 0
    buf: list[tuple[str, int]] = []
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            yield pl.DataFrame(buf, orient='row', schema=schema), idx
            idx += 1
            buf = []
    if buf:
        yield pl.DataFrame(buf, orient='row', schema=schema), idx


market_table = Table(
    'market',
    'static/market',
    (
        ('time', pl.Datetime(time_unit='ms', time_zone='UTC')),
        ('open', pl.Float64),
        ('high', pl.Float64),
        ('low', pl.Float64),
        ('close', pl.Float64),
    ),
    datadir=Path(__file__).parent.parent / 'tests/.test-db',
    partitioning=TimePartitionMeta(on_column='time'),
)


def market_frame_stream(
    *,
    start: datetime = epoch,
    periods: int = 10,
    step: timedelta = timedelta(minutes=1),
    batch_size: int = 1_000,
    seed: int = 42,
    start_price: float = 100.0,
    vol: float = 0.75,
) -> Generator[tuple[pl.DataFrame, int], None, None]:
    '''
    Stream synthetic OHLC bars for the *market_table* schema in batches.

    - Time starts at `start` (timezone-aware recommended) and increments by `step`.
    - Prices follow a simple seeded random walk around `start_price`.
    - Ensures OHLC invariants: high >= max(open, close), low <= min(open, close).

    Yields: (DataFrame, frame_index)

    '''
    rnd = random.Random(seed)

    # build rows first for deterministic batching
    rows = []
    last_close = start_price
    t = start
    for _ in range(periods):
        drift = rnd.gauss(0.0, vol)
        o = last_close
        c = max(0.01, o + drift)  # avoid going negative/zero
        wiggle = abs(rnd.gauss(0.0, vol))
        hi = max(o, c) + wiggle
        lo = max(0.0001, min(o, c) - wiggle)
        rows.append((t, float(o), float(hi), float(lo), float(c)))
        last_close = c
        t = t + step

    schema = market_table.schema.as_polars()

    # batch
    buf = []
    idx = 0
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            yield pl.DataFrame(buf, orient="row", schema=schema), idx
            idx += 1
            buf = []
    if buf:
        yield pl.DataFrame(buf, orient="row", schema=schema), idx
