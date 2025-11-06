import re
from pathlib import Path

import pytest
import polars as pl

from native_db._ctx import Context, ContextBuilder, open_ctx_writer
from native_db.dtypes import Keyword, Mono, TypeHints
from native_db.table import Table
from native_db.table._layout import DictionaryPartition, MonoPartition
from native_db.table.writer import TableWriterOptions

from native_db._testing import (
    BlockchainContext,
    block_frame_stream,
    blockchain_ctx,
)


def _mk_ctx(
    tmp_path: Path,
    row_size: int,
    rows_per_file: int,
    commit_threshold: int
) -> BlockchainContext:
    ctx = blockchain_ctx(tmp_path)
    t = ctx.blocks.copy(
        datadir=tmp_path,
        partitioning=MonoPartition(on_column='number', row_size=row_size),
        writer_opts=TableWriterOptions(
            commit_threshold=commit_threshold,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        )
    )
    return Context(
        datadir=tmp_path,
        tables=(t,),
        transforms=tuple(),
    )  # type: ignore


async def test_bucket_file_naming_and_indices(tmp_path, anyio_backend):
    row_size = 10_000
    commit_threshold = 5_000
    rows_per_file = 5_000
    ctx = _mk_ctx(tmp_path, row_size, rows_per_file, commit_threshold)
    builder = ContextBuilder(ctx)

    # Write 3 buckets worth of rows => buckets 0,1,2
    for rows, _ in block_frame_stream(
        start_number=0, end_number=30_000, batch_size=rows_per_file
    ):
        builder.extend('blocks', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

    # Check each bucket
    for b in (0, 1, 2):
        bdir = ctx.blocks.local_path / f'bucket={b}'
        files = sorted(p.name for p in bdir.glob(ctx.blocks.file_pattern))
        assert files, f'no parquet files under {bdir}'
        assert all(re.fullmatch(r'\d{5}\.parquet', f) for f in files)
        # indices are contiguous 00000..N
        seen = [int(f.split('.')[0]) for f in files]
        assert seen == list(range(len(seen))), (
            f'non-contiguous indices in {bdir}: {seen}'
        )


async def test_per_file_rowcount_and_key_bounds(tmp_path, anyio_backend):
    row_size = 10_000
    commit_threshold = 10_000
    rows_per_file = 2_000
    ctx = _mk_ctx(tmp_path, row_size, rows_per_file, commit_threshold)
    builder = ContextBuilder(ctx)

    # Write 3 buckets worth of rows => buckets 0,1,2
    for rows, _ in block_frame_stream(
        start_number=0, end_number=15_000, batch_size=rows_per_file
    ):
        builder.extend('blocks', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

    # Scan every written parquet file and assert counts + min/max in-range
    for bdir in sorted(ctx.blocks.local_path.glob('bucket=*')):
        b = int(bdir.name.split('=')[1])
        lo, hi_excl = b * commit_threshold, (b + 1) * commit_threshold
        for f in sorted(bdir.glob('part-*.parquet')):
            df = pl.read_parquet(f, columns=['number', 'bucket'])
            assert df.height == rows_per_file, (
                f'{f} has {df.height}, expected {rows_per_file}'
            )
            nmin, nmax = df['number'].min(), df['number'].max()
            assert lo <= nmin < hi_excl and lo <= nmax < hi_excl, (
                f'bounds out of bucket range: '
                f'{f} => [{nmin}, {nmax}] not within [{lo}, {hi_excl})'
            )
            # partition key column is present and correct (sink uses include_key)
            assert 'bucket' in df.columns and (
                df['bucket'].unique().to_list() == [b]
            )



async def test_hive_filter_by_bucket_scan(tmp_path, anyio_backend):
    row_size = 10_000
    commit_threshold = 5_000
    rows_per_file = 5_000
    ctx = _mk_ctx(tmp_path, row_size, rows_per_file, commit_threshold)
    builder = ContextBuilder(ctx)

    # Write 3 buckets worth of rows => buckets 0,1,2
    for rows, _ in block_frame_stream(
        start_number=0, end_number=50_000, batch_size=rows_per_file
    ):
        builder.extend('blocks', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

    # pick a bucket and verify
    b = 3
    lf = pl.scan_parquet(str((ctx.blocks.local_path / f'bucket={b}' / '*.parquet')))
    got = lf.select(pl.len()).collect().item()
    assert got == row_size, f'expected {row_size} rows in bucket {b}, got {got}'


def _rows_in_dir(root: Path) -> int:
    # Sum all parquet rows under a directory (recursive).
    scan = pl.scan_parquet(str(root / '**/*.parquet'))
    return 0 if scan is None else (scan.select(pl.len()).collect().item() or 0)


def _rows_per_bucket(root: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    if not root.exists():
        return out
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith('bucket='):
            bucket = int(child.name.split('=', 1)[1])
            out[bucket] = _rows_in_dir(child)
    return out


@pytest.mark.parametrize('rows_per_file, row_size', [(10_000, 10_000)])
async def test_stream_dense_blocks_partitioned_by_bucket(
    tmp_path, rows_per_file: int, row_size: int, anyio_backend
) -> None:
    '''
    Dense case:
      - numbers: 0..49,999
      - row_size (bucket width) = 10,000  ->  5 buckets: 0..4
      - rows_per_file = 10,000 so we write exactly five files overall (1 per bucket or more, but counts match).
    '''
    ctx = _mk_ctx(tmp_path, row_size, rows_per_file, rows_per_file)
    builder = ContextBuilder(ctx)

    # Emit 50k rows in 5 batches of 10k, strictly ordered
    total_rows = 50_000
    batches = block_frame_stream(
        start_number=0,
        end_number=total_rows,  # exclusive in helper
        batch_size=rows_per_file,
        order='ordered',
    )

    for rows, _ in batches:
        builder.extend('blocks', rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)

    # Validate partition layout and counts
    buckets = _rows_per_bucket(ctx.blocks.local_path)
    # Expect 5 buckets: 0..4 and each has exactly 10,000 rows
    expected = {b: rows_per_file for b in range(total_rows // row_size)}
    assert buckets == expected, f'Unexpected per-bucket rows: {buckets!r}'


def make_unpartitioned_ctx(tmp_path: Path) -> Context:
    return Context(
        datadir=tmp_path,
        tables=(
            Table(
                name="users",
                source="static/users",
                schema=(
                    ("id", pl.Int64),
                    ("name", pl.String, TypeHints(avg_str_size=8)),
                ),
                partitioning=None,
                compression="zstd",
            ),
        ),
        transforms=tuple(),
    )

def make_mono_partitioned_ctx(tmp_path: Path) -> Context:
    # small row_size so multiple buckets are created with a few rows
    return Context(
        datadir=tmp_path,
        tables=(
            Table(
                name="events",
                source="static/events",
                schema=(
                    ("eid", Mono(size=4), TypeHints(sort="asc")),  # sorted ascending
                    ("payload", pl.String, TypeHints(avg_str_size=16)),
                ),
                partitioning=MonoPartition(on_column="eid", row_size=3),
                compression="zstd",
            ),
        ),
        transforms=tuple(),
    )

async def drain_frame(ctx: Context, table: str, rows: list[tuple]) -> None:
    """
    Use TableBuilder to create a single in‑memory IPC buffer with all rows.
    Sorting and non‑null checks happen at flush-time per schema hints.
    """

    builder = ContextBuilder(ctx)
    builder.extend(table, rows)

    staged = await builder.stage(drain=True)

    async with open_ctx_writer(ctx) as writer:
        for table_name, frames in staged.items():
            for frame in frames:
                await writer.stage_direct(table_name, frame)


async def test_scan_roundtrip_unpartitioned(tmp_path: Path, anyio_backend):
    ctx = make_unpartitioned_ctx(tmp_path)

    rows = [
        (1, "ana"),
        (2, "bob"),
        (3, "cyd"),
    ]
    await drain_frame(ctx, 'users', rows)

    # 1) table.scan() must see the rows we wrote
    out_rows = (
        ctx.users.scan()
        .collect()
        .to_dicts()
    )
    assert len(out_rows) == len(rows)
    assert {(r["id"], r["name"]) for r in out_rows} == set(rows)

    # 2) file layout sanity (single part file created)
    assert len(ctx.users.files()) == 1  # writer places NNNNN.parquet on final root when no partitioning


async def test_scan_roundtrip_mono_partitioned(tmp_path: Path, anyio_backend):
    ctx = make_mono_partitioned_ctx(tmp_path)

    # eid row_size = 3 ⇒ buckets: 0 for 0..2, 1 for 3..5, 2 for 6..8, ...
    rows = [(0, "a"), (5, "b"), (2, "c"), (3, "d"), (8, "e"), (1, "f"), (6, "g")]
    await drain_frame(ctx, 'events', rows)

    # scan must read via hive partitioning when table has a partitioner
    lf = ctx.events.scan()
    df = lf.collect()

    # 1) all rows preserved
    assert df.height == len(rows)

    # # 2) per-partition sort by the partition column is enforced at sink time
    # # (writer uses per_partition_sort_by=self._part.plcol for partitioned sinks)
    # assert df["eid"].to_list() == sorted(r[0] for r in rows)

    # 3) buckets exist on disk (hive keys)
    # writer prepares 'bucket' from eid // row_size, includes hive key in output
    bucket_dirs = [p for p in (ctx.events.local_path).iterdir() if p.is_dir() and p.name.startswith("bucket=")]
    assert bucket_dirs, "expected hive bucket=... directories"
    # verify a couple of expected buckets (0,1,2) present for our eids
    bucket_names = {bd.name for bd in bucket_dirs}
    assert {"bucket=0", "bucket=1", "bucket=2"} <= bucket_names

def test_scan_empty_when_no_files(tmp_path: Path):
    # brand‑new table with no files on disk
    ctx = make_unpartitioned_ctx(tmp_path)
    # With local source and missing path, scan() returns an empty LazyFrame
    df = ctx.users.scan().collect()
    assert df.height == 0


async def test_scan_roundtrip_dictionary_string_depth2_lowercase(tmp_path: Path, anyio_backend):
    # Partition on first two letters of 'name', lowercased → char0, char1
    ctx = Context(
        datadir=tmp_path,
        tables=(
            Table(
                name="authors",
                source="static/authors",
                schema=(
                    ("name", Keyword, TypeHints(avg_str_size=8)),
                    ("pmid", Mono(size=8), TypeHints(sort="asc")),
                ),
                partitioning=DictionaryPartition(on_column="name", depth=2, lowercase=True),
                compression="zstd",
            ),
        ),
        transforms=tuple(),
    )

    rows = [
        ("Alice", 1),
        ("AL", 2),
        ("bob", 3),
        ("Bo", 4),
        ("charlie", 5),
        ("CH", 6),
    ]
    await drain_frame(ctx, 'authors', rows)

    # 1) What scan() sees should match what we wrote
    df = ctx.authors.scan().collect()  # uses hive_partitioning=True when table has a partitioner
    assert df.height == len(rows)
    # pmid sorted ASC within partitions due to per_partition_sort_by=self._part.plcol
    assert df["pmid"].to_list() == sorted(r[1] for r in rows)

    # 2) Hive dirs materialized as char0/char1 derived from lowercase(name)[:2]
    # Expected buckets: al, bo, ch
    root = ctx.authors.local_path
    for c0, c1 in [("a", "l"), ("b", "o"), ("c", "h")]:
        part_dir = root / f"char0={c0}" / f"char1={c1}"
        assert part_dir.is_dir(), f"missing {part_dir}"
        assert any(part_dir.glob(ctx.authors.file_pattern)), f"no parts under {part_dir}"

async def test_scan_roundtrip_dictionary_numeric_pad_depth2(tmp_path: Path, anyio_backend):
    # Partition on first two digits of zero-padded 'code' (width=4) → stable MSD buckets
    ctx = Context(
        datadir=tmp_path,
        tables=(
            Table(
                name="codes",
                source="static/codes",
                schema=(
                    ("code", Mono(size=8)),      # numeric id; we only partition by its string form
                    ("val", pl.String, TypeHints(avg_str_size=4)),
                ),
                partitioning=DictionaryPartition(on_column="code", depth=2, pad_to=4, signed=False),
                compression="zstd",
            ),
        ),
        transforms=tuple()
    )

    rows = [
        (7, "a"),     # "0007" → char0=0, char1=0
        (42, "b"),    # "0042" → char0=0, char1=0
        (105, "c"),   # "0105" → char0=0, char1=1
        (1002, "d"),  # "1002" → char0=1, char1=0
    ]
    await drain_frame(ctx, 'codes', rows)

    # 1) round-trip via scan()
    df = ctx.codes.scan().collect()
    assert df.height == len(rows)
    # verify all codes came back
    assert set(df["code"].to_list()) == {r[0] for r in rows}

    # 2) expected hive layout exists
    root = ctx.codes.local_path
    expect_dirs = [
        ("0", "0"),  # for 0007, 0042
        ("0", "1"),  # for 0105
        ("1", "0"),  # for 1002
    ]
    for c0, c1 in expect_dirs:
        part_dir = root / f"char0={c0}" / f"char1={c1}"
        assert part_dir.is_dir(), f"missing {part_dir}"
        assert any(part_dir.glob(ctx.codes.file_pattern)), f"no parts under {part_dir}"
