from datetime import date, datetime, timedelta
from pathlib import Path
import random
from typing import Generator, Literal, Protocol

import polars as pl

from native_db._ctx import Context
from native_db._utils import epoch
from native_db.dtypes import Keyword, Mono, TypeHints
from native_db.table import Table
from native_db.table._layout import DictionaryPartition, MonoPartition, TimePartition
from native_db.table.writer import TableWriterOptions
from native_db.transform import Transform


class BlockchainContext(Protocol):
    tables: tuple[Table, ...]
    transforms: tuple[Transform, ...]

    # tables
    blocks: Table

    def ensure_cache(self, regen: bool = False) -> None: ...


def blockchain_ctx(datadir: Path) -> BlockchainContext:
    ctx = Context(
        datadir=datadir,
        tables=(
            Table(
                'blocks',
                'static/blocks',
                (
                    ('number', Mono(size=4), TypeHints(sort='asc')),
                    ('timestamp', pl.Datetime(time_unit='us', time_zone='UTC')),
                    ('hash', Keyword, TypeHints(avg_str_size=64)),
                ),
                partitioning=MonoPartition(on_column='number')
            ),
        ),
        transforms=()
    )

    return ctx  # type: ignore


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
) -> Generator[tuple[list, int], None, None]:
    '''
    Yield batches as (DataFrame, frame_index).
    - order='ordered': emit in natural increasing frame order.
    - order='out_of_order': emit in shuffled order (optionally bounded).

    '''
    # 1. Build all batches
    rows_iter = block_stream(start_number, end_number, start_date, time_step)
    batches: list[tuple[list, int]] = []
    idx = 0
    buf = []
    for r in rows_iter:
        buf.append(r)
        if len(buf) == batch_size:
            batches.append((buf, idx))
            idx += 1
            buf = []
    if buf:
        batches.append((buf, idx))

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

class PubmedContext(Protocol):
    tables: tuple[Table, ...]
    transforms: tuple[Transform, ...]

    # tables
    pubmed: Table
    authors: Table
    articles: Table
    references: Table
    # transforms
    article_citations: Transform

    def ensure_cache(self, regen: bool = False) -> None: ...

def article_citations_cache_init(ctx: PubmedContext) -> pl.LazyFrame:
    max_b_cached = 0
    if ctx.article_citations.cache_path.exists():
        max_b_cached = (
            ctx.article_citations.scan(ctx)
            .select('bucket_id')
            .max()
            .collect()
            .item()
        )

    return (
        ctx.references.scan(use_cache=False)
        .group_by("cited_pmid")
        .agg(
            pl.len().alias('n_cites'),
            pl.col('bucket_id').max().alias('bucket_id'),
        )
        .rename({'cited_pmid': 'pmid'})
        .filter(pl.col('bucket_id').gt(max_b_cached))
        .sort('pmid')
    )

def article_citations_cache_checker(ctx: PubmedContext | None = None) -> bool:
    if ctx is None:
        return False

    if not ctx.article_citations.cache_path.exists():
        return False

    return (
        ctx.article_citations
        .scan(ctx)
        .select(pl.col('bucket_id').max())
        .collect()
        .item()
    ) == (
        ctx.references
        .scan(use_cache=False)
        .select(pl.col('bucket_id').max())
        .collect()
        .item()
    )

def pubmed_ctx(datadir: Path) -> PubmedContext:
    cache_dir = datadir / 'cache'
    ctx = Context(
        datadir=datadir,
        tables=(
            Table(
                'articles',
                'static/pubmed',
                (
                    ('pmid', Mono(size=8), TypeHints(sort='asc')),
                    ('pub_date', pl.Date),
                    ('title', pl.String, TypeHints(avg_str_size=128)),
                ),
                partitioning=MonoPartition(on_column='pmid')
            ),
            Table(
                'authors',
                'static/pubmed_author',
                (
                    ('name', Keyword),
                    ('pmid', Mono(size=8), TypeHints(sort='asc')),
                ),
                partitioning=DictionaryPartition(on_column='name', depth=1),
                writer_opts=TableWriterOptions(commit_threshold=100, rows_per_file=50)
            ),
            Table(
                name='references',
                source='static/references',
                schema=(
                    ('cited_pmid', Mono(size=8), TypeHints(sort='asc')),
                    ('citing_pmid', Mono(size=8), TypeHints(sort='asc')),
                    ('bucket_id',  pl.UInt32),
                ),
                format='parquet',
                partitioning=MonoPartition(
                    on_column='cited_pmid', row_size=1_000_000
                ),
                writer_opts=TableWriterOptions(
                    commit_threshold=100, rows_per_file=50
               ),
            ),
        ),
        transforms=(
            (
                # materialize number of citations per article
                # (reverse citations graph edges)
                Transform(
                    name='article_citations',
                    query=article_citations_cache_init,
                    cache_path=cache_dir
                    / 'article_citations.parquet',
                    is_cached=article_citations_cache_checker,
                    primary_key='pmid',
                )
            ),
        ),
    )

    return ctx  # type: ignore


def pubmed_frame_stream(
    *,
    # list of (start, end) inclusive ranges, emitted in order; gaps may exist between ranges
    ranges: list[tuple[int, int]],
    batch_size: int,
) -> Generator[tuple[list, int], None, None]:
    '''
    Yield (DataFrame, frame_index) batches representing a set of ordered pmid ranges,
    preserving local order and allowing gaps across ranges.

    - The frame_index is strictly increasing (0..N-1).
    - Titles are synthetic; pub_date increments daily for determinism.

    '''
    rows: list[tuple[int, date, str]] = []

    # deterministic dates/titles
    cur_date = epoch.date()
    for lo, hi in ranges:
        for pmid in range(lo, hi + 1):
            rows.append((pmid, cur_date, f'title-{pmid}'))
            cur_date = cur_date + timedelta(days=1)

    batches: list[tuple[list, int]] = []
    idx = 0
    buf: list[tuple[int, date, str]] = []
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            batches.append((buf, idx))
            idx += 1
            buf = []
    if buf:
        batches.append((buf, idx))

    for b in batches:
        yield b


def pubmed_author_random_frame_stream(
    *,
    total_rows: int,
    batch_size: int,
    seed: int = 123,
    authors: list[str] | None = None,
    pmid_start: int = 1,
) -> Generator[tuple[list, int], None, None]:
    """
    Yield (DataFrame, frame_index) batches matching pubmed_author_table's schema.

    - author_name: sampled from `authors` (or a default list).
    - pmid: increasing Mono32-compatible ints starting at pmid_start.
    """
    rng = random.Random(seed)
    if not authors:
        # keep ASCII-ish to avoid unicode slicing surprises in tests
        authors = ['Alice', 'Bob', 'Charlie', 'Dan', 'Eve', 'Mallory', 'Oscar']

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
            yield buf, idx
            idx += 1
            buf = []
    if buf:
        yield buf, idx


class MarketContext(Protocol):
    tables: tuple[Table, ...]
    transforms: tuple[Transform, ...]

    # tables
    market: Table

    def ensure_cache(self, regen: bool = False) -> None: ...


def market_ctx(datadir: Path) -> MarketContext:
    ctx = Context(
        datadir=datadir,
        tables=(
            Table(
                'market',
                'static/market',
                (
                    ('time', pl.Datetime(time_unit='ms', time_zone='UTC')),
                    ('open', pl.Float64),
                    ('high', pl.Float64),
                    ('low', pl.Float64),
                    ('close', pl.Float64),
                ),
                partitioning=TimePartition(on_column='time'),
            ),
        ),
        transforms=()
    )

    return ctx  # type: ignore


def market_frame_stream(
    *,
    start: datetime = epoch,
    periods: int = 10,
    step: timedelta = timedelta(minutes=1),
    batch_size: int = 1_000,
    seed: int = 42,
    start_price: float = 100.0,
    vol: float = 0.75,
) -> Generator[tuple[list, int], None, None]:
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

    # batch
    buf = []
    idx = 0
    for r in rows:
        buf.append(r)
        if len(buf) == batch_size:
            yield buf, idx
            idx += 1
            buf = []
    if buf:
        yield buf, idx
