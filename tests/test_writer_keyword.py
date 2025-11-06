from pathlib import Path

import polars as pl

from native_db._ctx import Context, ContextBuilder, open_ctx_writer
from native_db._testing import (
    pubmed_ctx,
    pubmed_author_random_frame_stream,
)
from native_db.table import DictionaryPartitioner, DictionaryPartition
from native_db.table.writer import TableWriter, TableWriterOptions


def test_keyword_partitioner_by_cols_depth1(tmp_path):
    ctx = pubmed_ctx(tmp_path)
    # default table uses char_depth=1 in helper
    part = ctx.authors.partitioning
    assert isinstance(part, DictionaryPartitioner)
    assert part.by_cols == ['char0']


def test_keyword_partitioner_prepare_depth1_basic(tmp_path):
    ctx = pubmed_ctx(tmp_path)
    # simple ASCII names to avoid unicode edge-cases in byte vs. char slicing
    df = pl.DataFrame(
        {'name': ['Alice', 'Bob', 'Charlie', '', 'Zed'],
         'pmid': [1, 2, 3, 4, 5]},
        schema=ctx.authors.schema.as_polars(),
    )
    assert ctx.authors.partitioning
    out = ctx.authors.partitioning.prepare(df.lazy()).collect()
    assert 'char0' in out.columns
    # first character or '' for empty
    assert out['char0'].to_list() == ['A', 'B', 'C', '', 'Z']


def test_keyword_partitioner_by_cols_depth2(tmp_path):
    ctx = pubmed_ctx(tmp_path)
    # Same schema, but force char_depth=2
    t2 = ctx.authors.copy(
        partitioning=DictionaryPartition(on_column='name', depth=2),
        name='pubmed_author_depth2',
        source='static/pubmed_author_depth2',
    )
    part = t2.partitioning
    assert isinstance(part, DictionaryPartitioner)
    assert part.by_cols == ['char0', 'char1']

    df = pl.DataFrame(
        {'name': ['Amy', 'Bo', 'C', ''],
         'pmid': [10, 11, 12, 13]},
        schema=t2.schema.as_polars(),
    )
    out = part.prepare(df.lazy()).collect()
    assert out['char0'].to_list() == ['A', 'B', 'C', '']
    # second character (or '' if not present)
    assert out['char1'].to_list() == ['m', 'o', '', '']


async def test_writer_creates_keyword_partitions(tmp_path: Path, anyio_backend):
    """
    Integration: make sure differently named authors land under different
    char buckets on disk (char0=A, char0=B, ...).
    """
    ctx = pubmed_ctx(tmp_path)

    # Emit two batches with clearly different first letters
    # e.g., 6 'Alice*' rows and 6 'Bob*' rows -> two buckets 'A' and 'B'
    # (batch_size > rows_per_file to ensure at least one file per bucket)
    batches = list(
        pubmed_author_random_frame_stream(
            total_rows=100,
            batch_size=100,  # single batch for determinism
            seed=7,
            authors=['Alice', 'Bob'],
            pmid_start=1000,
        )
    )

    builder = ContextBuilder(ctx)
    for rows, i in batches:
        builder.extend('authors', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)


    # force a final commit if needed by pushing an empty no-op with finish semantics
    # (TableWriter commits automatically if threshold met; ensure it happened)
    # There isn't a "finish()" hook, but we can check the directory.
    root = ctx.authors.local_path
    assert root.exists(), "table root should exist after push"

    # Expect hive-style partition dirs like char0=A and char0=B
    buckets = {p.name for p in root.iterdir() if p.is_dir() and p.name != '.staging'}
    assert 'char0=A' in buckets
    assert 'char0=B' in buckets

    # And each bucket should have at least one parquet
    for b in ('char0=A', 'char0=B'):
        parts = sorted((root / b).glob(ctx.authors.file_pattern))
        assert parts, f'missing parquet parts under {b}'


async def test_writer_creates_keyword_partitions_depth2(tmp_path: Path, anyio_backend):
    """
    Integration: make sure differently named authors land under different
    char buckets on disk (char0=<X>, char1=<Y>).
    """
    ctx = pubmed_ctx(tmp_path)

    ctx = Context(
        datadir=tmp_path,
        tables=(
            # Override to depth=2
            ctx.authors.copy(
                source='static/pubmed_author_it2',
                partitioning=DictionaryPartition(on_column='name', depth=2),
                writer_opts=TableWriterOptions(commit_threshold=100, rows_per_file=50)
            ),
        ),
        transforms=()
    )

    # Two batches with authors that differ on both first and second char
    batches = list(
        pubmed_author_random_frame_stream(
            total_rows=100,
            batch_size=100,  # single batch for determinism
            seed=7,
            authors=['Al', 'Bo'],   # ensures char0=A/B and char1=l/o
            pmid_start=1000,
        )
    )

    builder = ContextBuilder(ctx)
    for rows, i in batches:
        builder.extend('authors', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

    root = ctx.authors.local_path
    assert root.exists(), "table root should exist after push"

    # Expect hive-style partition dirs with both char0 and char1 keys
    buckets = {p.name for p in root.iterdir() if p.is_dir() and p.name != '.staging'}
    assert 'char0=A' in buckets or 'char0=B' in buckets

    # dig into second-level dirs
    for first in ('char0=A', 'char0=B'):
        if (root / first).exists():
            second_level = {p.name for p in (root / first).iterdir() if p.is_dir()}
            assert all(s.startswith('char1=') for s in second_level)
            for sec in second_level:
                parts = sorted((root / first / sec).glob(ctx.authors.file_pattern))
                assert parts, f'missing parquet parts under {first}/{sec}'
