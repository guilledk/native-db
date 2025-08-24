import re
from pathlib import Path

import pytest
import polars as pl

from native_db.table import Table
from native_db.table._layout import MonoPartitionMeta
from native_db.table.writer import TableWriter, TableWriterOptions

from native_db._testing import (
    block_frame_stream,
    block_table,
    pubmed_table,
    pubmed_frame_stream,
)


def test_bucket_file_naming_and_indices(tmp_path):
    row_size = 10_000
    rows_per_file = 5_000  # two files per bucket when we push 10k
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartitionMeta(on_column='number', row_size=row_size),
    )

    w = TableWriter(
        t,
        options=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        ),
    )

    # Write 3 buckets worth of rows => buckets 0,1,2
    for df, idx in block_frame_stream(
        start_number=0, end_number=30_000, batch_size=rows_per_file
    ):
        w.push(df, frame_index=idx, number_of_rows=df.height)

    # Check each bucket
    for b in (0, 1, 2):
        bdir = t.local_path / f'bucket={b}'
        files = sorted(p.name for p in bdir.glob('part-*.parquet'))
        assert files, f'no parquet files under {bdir}'
        assert all(re.fullmatch(r'part-\d{5}\.parquet', f) for f in files)
        # indices are contiguous 00000..N
        seen = [int(f.split('-')[1].split('.')[0]) for f in files]
        assert seen == list(range(len(seen))), (
            f'non-contiguous indices in {bdir}: {seen}'
        )


def test_per_file_rowcount_and_key_bounds(tmp_path):
    row_size = 10_000
    rows_per_file = 2_000
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartitionMeta(on_column='number', row_size=row_size),
    )
    w = TableWriter(
        t,
        options=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        ),
    )

    # Write 1.5 buckets (15k rows) in exact 2k frames + one 1k tail (which stays staged)
    for df, idx in block_frame_stream(
        start_number=0, end_number=15_000, batch_size=rows_per_file
    ):
        w.push(df, frame_index=idx, number_of_rows=df.height)

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


def test_out_of_order_frames_delay_commit(tmp_path):
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartitionMeta(on_column='number', row_size=10_000),
    )
    rows_per_file = 1_000
    w = TableWriter(
        t,
        options=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        ),
    )

    # Build three frames (0,1,2) but push 1,2 first (gap at 0)
    batches = list(
        block_frame_stream(
            start_number=0, end_number=3_000, batch_size=rows_per_file
        )
    )
    df0, idx0 = batches[0]
    for df, idx in batches[1:]:
        w.push(df, frame_index=idx, number_of_rows=df.height)

    # No files should exist yet (gap prevents commit)
    assert not list(t.local_path.rglob('part-*.parquet'))

    # Now push the missing first frame; commit should fire
    w.push(df0, frame_index=idx0, number_of_rows=df0.height)
    files = list(t.local_path.rglob('part-*.parquet'))
    assert files, 'commit did not happen after filling the gap'


def test_hive_filter_by_bucket_scan(tmp_path):
    row_size = 10_000
    rows_per_file = 5_000
    t = block_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartitionMeta(on_column='number', row_size=row_size),
    )
    w = TableWriter(
        t,
        options=TableWriterOptions(
            commit_threshold=rows_per_file,
            rows_per_file=rows_per_file,
            start_frame_index=0,
        ),
    )

    for df, idx in block_frame_stream(
        start_number=0, end_number=50_000, batch_size=rows_per_file
    ):
        w.push(df, frame_index=idx, number_of_rows=df.height)

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
def test_stream_dense_blocks_partitioned_by_bucket(
    tmp_path, rows_per_file: int, row_size: int
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
        partitioning=MonoPartitionMeta(on_column='number', row_size=row_size),
    )

    writer = TableWriter(
        t,
        options=TableWriterOptions(
            commit_threshold=rows_per_file,  # commit every batch
            rows_per_file=rows_per_file,
            start_frame_index=0,
        ),
    )

    # Emit 50k rows in 5 batches of 10k, strictly ordered
    total_rows = 50_000
    batches = block_frame_stream(
        start_number=0,
        end_number=total_rows,  # exclusive in helper
        batch_size=rows_per_file,
        order='ordered',
    )

    for df, idx in batches:
        # Provide frame_index to keep the commit path contiguous
        writer.push(df, frame_index=idx, number_of_rows=df.height)

    # Validate partition layout and counts
    buckets = _rows_per_bucket(t.local_path)
    # Expect 5 buckets: 0..4 and each has exactly 10,000 rows
    expected = {b: rows_per_file for b in range(total_rows // row_size)}
    assert buckets == expected, f'Unexpected per-bucket rows: {buckets!r}'


@pytest.mark.parametrize(
    'ranges, rows_per_file, row_size, expected',
    [
        # Gapped case:
        #   - range A: 0..4,999     -> bucket 0 (5,000 rows)
        #   - range B: 20,000..29,999 -> bucket 2 (10,000 rows)
        # With row_size=10,000, we expect buckets {0: 5k, 2: 10k}.
        ([(0, 4_999), (20_000, 29_999)], 5_000, 10_000, {0: 5_000, 2: 10_000}),
    ],
)
def test_stream_gapped_pubmed_partitioned_by_bucket(
    tmp_path,
    ranges: list[tuple[int, int]],
    rows_per_file: int,
    row_size: int,
    expected: dict[int, int],
) -> None:
    '''
    Gapped monotonic case (allow_gaps=True):
      - Use pubmed_frame_stream to generate disjoint pmid ranges
      - Partition by bucket computed as pmid // row_size (works regardless of gaps).
    '''
    t: Table = pubmed_table.copy(
        datadir=tmp_path,
        partitioning=MonoPartitionMeta(on_column='pmid', row_size=row_size),
    )

    writer = TableWriter(
        t,
        options=TableWriterOptions(
            commit_threshold=rows_per_file,  # commit every batch
            rows_per_file=rows_per_file,
            start_frame_index=0,
        ),
    )

    # Build frames with batch_size==rows_per_file for neat multiples
    batches = pubmed_frame_stream(ranges=ranges, batch_size=rows_per_file)
    for df, idx in batches:
        writer.push(df, frame_index=idx, number_of_rows=df.height)

    # validate written data
    root = Path(t.local_path)

    # 1) Compute expected rows per hive bucket from the gapped ranges
    #    (works even if a range crosses bucket boundaries).
    expected = {}  # bucket_id -> row_count
    for lo, hi in ranges:
        # ranges are inclusive in your tests, so use hi+1 as exclusive bound
        for b in range(lo // row_size, hi // row_size + 1):
            blo, bhi = b * row_size, (b + 1) * row_size
            overlap = max(0, min(hi + 1, bhi) - max(lo, blo))
            if overlap:
                expected[b] = expected.get(b, 0) + overlap

    # 2) Buckets present on disk should match exactly the expected set (gaps allowed)
    present_buckets = {int(p.name.split('=')[1]) for p in root.glob('bucket=*')}
    assert present_buckets == set(expected), (
        f'unexpected bucket dirs: {present_buckets} vs {set(expected)}'
    )

    # 3) Per-bucket file naming and contiguous local indices
    for b in sorted(present_buckets):
        bdir = root / f'bucket={b}'
        files = sorted(bdir.glob('part-*.parquet'))
        # indices contiguous: part-00000, part-00001, ...
        idxs = [int(f.stem.split('-')[1]) for f in files]
        assert idxs == list(range(len(idxs))), (
            f'non-contiguous part indices in {bdir}: {idxs}'
        )
        # each file has exactly rows_per_file rows
        for f in files:
            n = pl.read_parquet(f, columns=['pmid']).height
            assert n == rows_per_file, f'{f} has {n}, expected {rows_per_file}'

    # 4) Scan all written data; validate counts, bounds, uniqueness, and mapping
    lf = pl.scan_parquet(str(root / 'bucket=*/part-*.parquet'))

    # 4a) pmid -> bucket mapping is correct for every row
    ok = (
        lf.select(((pl.col('pmid') // row_size) == pl.col('bucket')).all())
        .collect()
        .item()
    )
    assert ok, 'pmid // row_size != bucket for some rows'

    # 4b) total rows written equals sum of full files (writer leaves remainder in staging)
    written_total = lf.select(pl.len()).collect().item()
    expected_written_total = sum(
        (c // rows_per_file) * rows_per_file for c in expected.values()
    )
    assert written_total == expected_written_total, (
        f'written_total={written_total} vs expected full-file rows={expected_written_total}'
    )

    # 4c) per-bucket row counts match (only full files accounted for)
    by_bucket = (
        lf.group_by('bucket')
        .agg(
            pl.len().alias('rows'),
            pl.col('pmid').min().alias('pmid_min'),
            pl.col('pmid').max().alias('pmid_max'),
        )
        .collect()
    )

    got = {
        int(r['bucket']): int(r['rows'])
        for r in by_bucket.iter_rows(named=True)
    }
    want = {
        b: (c // rows_per_file) * rows_per_file for b, c in expected.items()
    }
    assert got == want, f'per-bucket rows mismatch: got={got}, want={want}'

    # 4d) pmid min/max sit within each bucket’s range
    for r in by_bucket.iter_rows(named=True):
        b = int(r['bucket'])
        lo, hi_excl = b * row_size, (b + 1) * row_size
        assert (
            lo <= r['pmid_min'] < hi_excl and lo <= r['pmid_max'] < hi_excl
        ), (
            f'bucket {b} pmid bounds [{r["pmid_min"]}, {r["pmid_max"]}] not within [{lo}, {hi_excl})'
        )

    # 4e) no duplicate pmids among written rows
    n_unique = lf.select(pl.col('pmid').n_unique()).collect().item()
    assert n_unique == written_total, (
        f'found duplicates: n_unique={n_unique}, rows={written_total}'
    )

    # 5) Optional: exact file count per bucket (only full files)
    for b, total in expected.items():
        bdir = root / f'bucket={b}'
        files = list(bdir.glob('part-*.parquet'))
        assert len(files) == total // rows_per_file, (
            f'bucket {b} file count mismatch: {len(files)} vs {total // rows_per_file}'
        )
