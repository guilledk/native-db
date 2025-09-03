from io import BytesIO
import re
from pathlib import Path

import pytest
import polars as pl

from native_db.dtypes import Keyword, Mono, TypeHints
from native_db.table import Table
from native_db.table._layout import DictionaryPartition, MonoPartition
from native_db.table.writer import TableWriterOptions

from native_db._testing import (
    block_frame_stream,
    block_table,
)


async def test_bucket_file_naming_and_indices(tmp_path, anyio_backend):
    row_size = 10_000
    rows_per_file = 5_000  # two files per bucket when we push 10k
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartition(on_column='number', row_size=row_size),
        writer_opts=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        )
    )

    w = t.writer()
    # Write 3 buckets worth of rows => buckets 0,1,2
    for df, idx in block_frame_stream(
        start_number=0, end_number=30_000, batch_size=rows_per_file
    ):
        await w.stage_frame(df, index=idx)

    # Check each bucket
    for b in (0, 1, 2):
        bdir = t.local_path / f'bucket={b}'
        files = sorted(p.name for p in bdir.glob(t.file_pattern))
        assert files, f'no parquet files under {bdir}'
        assert all(re.fullmatch(r'\d{5}\.parquet', f) for f in files)
        # indices are contiguous 00000..N
        seen = [int(f.split('.')[0]) for f in files]
        assert seen == list(range(len(seen))), (
            f'non-contiguous indices in {bdir}: {seen}'
        )


async def test_per_file_rowcount_and_key_bounds(tmp_path, anyio_backend):
    row_size = 10_000
    rows_per_file = 2_000
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartition(on_column='number', row_size=row_size),
        writer_opts=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        )
    )

    w = t.writer()
    # Write 1.5 buckets (15k rows) in exact 2k frames + one 1k tail (which stays staged)
    for df, idx in block_frame_stream(
        start_number=0, end_number=15_000, batch_size=rows_per_file
    ):
        await w.stage_frame(df, index=idx)

    # Scan every written parquet file and assert counts + min/max in-range
    for bdir in sorted(t.local_path.glob('bucket=*')):
        b = int(bdir.name.split('=')[1])
        lo, hi_excl = b * row_size, (b + 1) * row_size
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


async def test_out_of_order_frames_delay_commit(tmp_path, anyio_backend):
    rows_per_file = 1_000
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartition(on_column='number', row_size=10_000),
        writer_opts=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        )
    )

    # Build three frames (0,1,2) but push 1,2 first (gap at 0)
    batches = list(
        block_frame_stream(
            start_number=0, end_number=3_000, batch_size=rows_per_file
        )
    )
    df0, idx0 = batches[0]

    w = t.writer()
    for df, idx in batches[1:]:
        await w.stage_frame(df, index=idx)

    # No files should exist yet (gap prevents commit)
    assert not t.files()

    # Now push the missing first frame; commit should fire
    await w.stage_frame(df0, index=idx0)
    assert t.files(), 'commit did not happen after filling the gap'


async def test_hive_filter_by_bucket_scan(tmp_path, anyio_backend):
    row_size = 10_000
    rows_per_file = 5_000
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartition(on_column='number', row_size=row_size),
        writer_opts=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        )
    )
    w = t.writer()
    for df, idx in block_frame_stream(
        start_number=0, end_number=50_000, batch_size=rows_per_file
    ):
        await w.stage_frame(df, index=idx)

    # pick a bucket and verify
    b = 3
    lf = pl.scan_parquet(str((t.local_path / f'bucket={b}' / '*.parquet')))
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
    # Use a copy of the table with a smaller bucket size for clearer layout
    t: Table = block_table.copy(
        datadir=tmp_path,
        # keep same source/datadir; change partitioning to smaller row_size
        partitioning=MonoPartition(on_column='number', row_size=row_size),
        writer_opts=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        )
    )

    # Emit 50k rows in 5 batches of 10k, strictly ordered
    total_rows = 50_000
    batches = block_frame_stream(
        start_number=0,
        end_number=total_rows,  # exclusive in helper
        batch_size=rows_per_file,
        order='ordered',
    )
    w = t.writer()
    for df, idx in batches:
        # Provide frame_index to keep the commit path contiguous
            await w.stage_frame(df, index=idx)

    # Validate partition layout and counts
    buckets = _rows_per_bucket(t.local_path)
    # Expect 5 buckets: 0..4 and each has exactly 10,000 rows
    expected = {b: rows_per_file for b in range(total_rows // row_size)}
    assert buckets == expected, f'Unexpected per-bucket rows: {buckets!r}'


def make_unpartitioned_table(tmp_path: Path) -> Table:
    return Table(
        name="users",
        source="static/users",
        schema=(
            ("id", pl.Int64),
            ("name", pl.String, TypeHints(avg_str_size=8)),
        ),
        datadir=tmp_path,
        partitioning=None,
        compression="zstd",
    )

def make_mono_partitioned_table(tmp_path: Path) -> Table:
    # small row_size so multiple buckets are created with a few rows
    return Table(
        name="events",
        source="static/events",
        schema=(
            ("eid", Mono(size=4), TypeHints(sort="asc")),  # sorted ascending
            ("payload", pl.String, TypeHints(avg_str_size=16)),
        ),
        datadir=tmp_path,
        partitioning=MonoPartition(on_column="eid", row_size=3),
        compression="zstd",
    )

def build_frame(table: Table, rows: list[tuple]) -> pl.DataFrame:
    """
    Use TableBuilder to create a single in‑memory IPC buffer with all rows.
    Sorting and non‑null checks happen at flush-time per schema hints.
    """
    tb = table.builder()
    tb.extend(rows)
    return tb.flush_frame()


async def test_scan_roundtrip_unpartitioned(tmp_path: Path, anyio_backend):
    t = make_unpartitioned_table(tmp_path)

    rows = [
        (1, "ana"),
        (2, "bob"),
        (3, "cyd"),
    ]
    frame = build_frame(t, rows)

    # stream to disk through TableWriter (staging -> commit)
    w = t.writer()
    await w.stage_frame(frame, index=0)
    await w.drain()

    # 1) table.scan() must see the rows we wrote
    out_rows = (
        t.scan()
        .collect()
        .to_dicts()
    )
    assert len(out_rows) == len(rows)
    assert {(r["id"], r["name"]) for r in out_rows} == set(rows)

    # 2) file layout sanity (single part file created)
    assert len(t.files()) == 1  # writer places NNNNN.parquet on final root when no partitioning


async def test_scan_roundtrip_mono_partitioned(tmp_path: Path, anyio_backend):
    t = make_mono_partitioned_table(tmp_path)

    # eid row_size = 3 ⇒ buckets: 0 for 0..2, 1 for 3..5, 2 for 6..8, ...
    rows = [(0, "a"), (5, "b"), (2, "c"), (3, "d"), (8, "e"), (1, "f"), (6, "g")]
    frame = build_frame(t, rows)

    w = t.writer()
    await w.stage_frame(frame, index=0)
    await w.drain()

    # scan must read via hive partitioning when table has a partitioner
    lf = t.scan()
    df = lf.collect()

    # 1) all rows preserved
    assert df.height == len(rows)

    # # 2) per-partition sort by the partition column is enforced at sink time
    # # (writer uses per_partition_sort_by=self._part.plcol for partitioned sinks)
    # assert df["eid"].to_list() == sorted(r[0] for r in rows)

    # 3) buckets exist on disk (hive keys)
    # writer prepares 'bucket' from eid // row_size, includes hive key in output
    bucket_dirs = [p for p in (t.local_path).iterdir() if p.is_dir() and p.name.startswith("bucket=")]
    assert bucket_dirs, "expected hive bucket=... directories"
    # verify a couple of expected buckets (0,1,2) present for our eids
    bucket_names = {bd.name for bd in bucket_dirs}
    assert {"bucket=0", "bucket=1", "bucket=2"} <= bucket_names

def test_scan_empty_when_no_files(tmp_path: Path):
    # brand‑new table with no files on disk
    t = make_unpartitioned_table(tmp_path)
    # With local source and missing path, scan() returns an empty LazyFrame
    df = t.scan().collect()
    assert df.height == 0

async def test_builder_validations_and_partial_flush(tmp_path: Path, anyio_backend):
    # Demonstrate builder sort/non-null checks and partial-buffer semantics
    t = Table(
        name="checks",
        source="static/checks",
        schema=(
            ("k", Mono(size=4), TypeHints(sort="asc")),  # sorted ascending
            ("v", pl.String, TypeHints(avg_str_size=4)), # non-null by default
        ),
        datadir=tmp_path,
    )

    tb = t.builder(target_row_size=3)  # require exactly 3 rows on flush
    tb.append((2, "b"))
    tb.append((1, "a"))
    tb.append((3, "c"))
    # flush enforces exact target size and sorts by hints before writing IPC
    frame = tb.flush_frame()
    w = t.writer()

    await w.stage_frame(frame, index=0)
    await w.drain()

    # # table is sorted by 'k' because builder sorted emitted slice; writer (no partitioner) preserves that order on disk
    # out_k = t.scan().select("k").collect().to_series().to_list()
    # assert out_k == [1, 2, 3]

    # any remaining rows in the builder (if we had overfilled) would be kept for next flush


async def test_scan_roundtrip_dictionary_string_depth2_lowercase(tmp_path: Path, anyio_backend):
    # Partition on first two letters of 'name', lowercased → char0, char1
    t = Table(
        name="authors",
        source="static/authors",
        schema=(
            ("name", Keyword, TypeHints(avg_str_size=8)),
            ("pmid", Mono(size=8), TypeHints(sort="asc")),
        ),
        datadir=tmp_path,
        partitioning=DictionaryPartition(on_column="name", depth=2, lowercase=True),
        compression="zstd",
    )

    rows = [
        ("Alice", 1),
        ("AL", 2),
        ("bob", 3),
        ("Bo", 4),
        ("charlie", 5),
        ("CH", 6),
    ]
    frame = build_frame(t, rows)
    w = t.writer()
    await w.stage_frame(frame, index=0)
    await w.drain()

    # 1) What scan() sees should match what we wrote
    df = t.scan().collect()  # uses hive_partitioning=True when table has a partitioner
    assert df.height == len(rows)
    # pmid sorted ASC within partitions due to per_partition_sort_by=self._part.plcol
    assert df["pmid"].to_list() == sorted(r[1] for r in rows)

    # 2) Hive dirs materialized as char0/char1 derived from lowercase(name)[:2]
    # Expected buckets: al, bo, ch
    root = t.local_path
    for c0, c1 in [("a", "l"), ("b", "o"), ("c", "h")]:
        part_dir = root / f"char0={c0}" / f"char1={c1}"
        assert part_dir.is_dir(), f"missing {part_dir}"
        assert any(part_dir.glob(t.file_pattern)), f"no parts under {part_dir}"

async def test_scan_roundtrip_dictionary_numeric_pad_depth2(tmp_path: Path, anyio_backend):
    # Partition on first two digits of zero-padded 'code' (width=4) → stable MSD buckets
    t = Table(
        name="codes",
        source="static/codes",
        schema=(
            ("code", Mono(size=8)),      # numeric id; we only partition by its string form
            ("val", pl.String, TypeHints(avg_str_size=4)),
        ),
        datadir=tmp_path,
        partitioning=DictionaryPartition(on_column="code", depth=2, pad_to=4, signed=False),
        compression="zstd",
    )

    rows = [
        (7, "a"),     # "0007" → char0=0, char1=0
        (42, "b"),    # "0042" → char0=0, char1=0
        (105, "c"),   # "0105" → char0=0, char1=1
        (1002, "d"),  # "1002" → char0=1, char1=0
    ]
    frame = build_frame(t, rows)
    w = t.writer()
    await w.stage_frame(frame, index=0)
    await w.drain()

    # 1) round-trip via scan()
    df = t.scan().collect()
    assert df.height == len(rows)
    # verify all codes came back
    assert set(df["code"].to_list()) == {r[0] for r in rows}

    # 2) expected hive layout exists
    root = t.local_path
    expect_dirs = [
        ("0", "0"),  # for 0007, 0042
        ("0", "1"),  # for 0105
        ("1", "0"),  # for 1002
    ]
    for c0, c1 in expect_dirs:
        part_dir = root / f"char0={c0}" / f"char1={c1}"
        assert part_dir.is_dir(), f"missing {part_dir}"
        assert any(part_dir.glob(t.file_pattern)), f"no parts under {part_dir}"
