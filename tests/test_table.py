from datetime import datetime, timezone
from io import BytesIO

import warnings
import polars as pl

from native_db.table import LayoutOptions
from native_db.table._layout import MonoPartOptions
from native_db.table.builder import TableBuilder
from native_db.table.writer import TableWriter

from native_db._testing import block_frame_stream, block_table, block_stream
from native_db._utils import NativeDBWarning


def test_definitions():
    print(block_table.pretty_str())
    struct_cls = block_table.struct
    assert isinstance(
        struct_cls(number=0, timestamp=datetime.now(tz=timezone.utc), hash='test'),
        block_table.struct
    )


def test_builder():
    builder = TableBuilder(block_table)

    rows = [row for row in block_stream()]
    builder.extend(rows)

    frame = pl.scan_ipc(BytesIO(builder.flush())).collect()

    assert frame.select(pl.len()).item() == len(rows)
    assert frame.schema == block_table.schema.as_polars()
    assert frame.rows() == rows




def test_writer(tmp_path):
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        table = block_table.copy(
            datadir=tmp_path,
            layout=LayoutOptions(
                rows_per_file=100,
                commit_threshold=80,  # will cause warning cause not divisible by rows_per_file
            )
        )

        assert any(isinstance(wi.message, NativeDBWarning) for wi in warns)

    w = TableWriter(table)

    # Build batches via generator (5 batches of 40 rows: 0..39, 40..79, ...)
    gen = block_frame_stream(
        start_number=0,
        end_number=200,
        batch_size=40,
    )
    batches_by_idx = {idx: df for (df, idx) in gen}

    # Out-of-order arrival, but pass correct frame_index each time
    arrival_order = [2, 0, 3, 1, 4]

    glob_exp = f'{table.prefix}-part-*.{table.suffix}'

    def count_parts():
        return len(list(table.local_path.glob(glob_exp)))


    before = count_parts()
    for i, idx in enumerate(arrival_order):
        w.push(batches_by_idx[idx], frame_index=idx)
        after = count_parts()

        if i < 3:
            assert after == before
        elif i == 3:
            assert after == before + 1
            before = after
            first_part = sorted(table.local_path.glob(glob_exp))[0]
            out = pl.read_parquet(first_part).select('number').to_series()
            assert out.len() == 100
            assert out.to_list() == list(range(0, 100))
        else:
            pass


    # Final state: two parts (200 rows), numbers 0..199 in order
    parts = sorted(table.local_path.glob(glob_exp))
    assert len(parts) == 2

    all_out = (
        pl.scan_parquet(table.local_path / glob_exp)
        .select('number')
        .collect()
        .to_series()
    )
    assert all_out.len() == 200
    assert all_out.sort().to_list() == list(range(0, 200))



def test_writer_partitioned(tmp_path):
    table = block_table.copy(
        datadir=tmp_path,
        layout=LayoutOptions(
            rows_per_file=100,
            commit_threshold=1000,  # commit in 1k-row chunks
            partitioning=MonoPartOptions(on_column='number', row_size=10_000)
        )
    )

    w = TableWriter(table)

    # stream 25k rows in 100-row batches
    for frame, _ in block_frame_stream(
        start_number=0,
        end_number=25_000,
        batch_size=100,
    ):
        w.push(frame)

    # layout checks
    root = table.local_path
    part_dirs = sorted(p for p in root.glob('*') if p.is_dir() and str(p.name).isnumeric())
    # Expect 3 partitions: two full (10k) and one half (5k)
    assert [d.name for d in part_dirs] == [
        '00000000', '00000001', '00000002'
    ]

    glob_exp = f'{table.prefix}-part-*.{table.suffix}'

    # Collect all parts in a stable order (partition path, then part filename)
    def parts_in(dirpath):
        return sorted(dirpath.glob(glob_exp))

    parts_0 = parts_in(part_dirs[0])
    parts_1 = parts_in(part_dirs[1])
    parts_2 = parts_in(part_dirs[2])

    # rows_per_file=100 -> 10k => 100 parts; 5k => 50 parts
    assert len(parts_0) == 100
    assert len(parts_1) == 100
    assert len(parts_2) == 50

    # per-file checks (row count + min/max correctness and contiguity)
    # Global expectation: part i should cover [i*100 .. i*100+99]
    all_parts = parts_0 + parts_1 + parts_2
    for i, p in enumerate(all_parts):
        s = pl.read_parquet(p, columns=['number']).to_series()
        assert s.len() == 100
        mn, mx = int(s.min()), int(s.max())
        exp_mn = i * 100
        exp_mx = exp_mn + 99
        assert mn == exp_mn and mx == exp_mx, f'{p} has [{mn},{mx}] != [{exp_mn},{exp_mx}]'

    # per-partition sanity: totals + min/max inside each partition
    # Partition id is derived from committed rows // 10_000, so ranges are:
    #   pid 0: [0 .. 9_999], pid 1: [10_000 .. 19_999], pid 2: [20_000 .. 24_999]
    def partition_stats(dirpath):
        lf = pl.scan_parquet(dirpath / glob_exp)
        out = lf.select(
            pl.len().alias('n'),
            pl.min('number').alias('mn'),
            pl.max('number').alias('mx'),
        ).collect().row(0)
        return int(out[0]), int(out[1]), int(out[2])

    n0, mn0, mx0 = partition_stats(part_dirs[0])
    n1, mn1, mx1 = partition_stats(part_dirs[1])
    n2, mn2, mx2 = partition_stats(part_dirs[2])

    assert (n0, mn0, mx0) == (10_000, 0, 9_999)
    assert (n1, mn1, mx1) == (10_000, 10_000, 19_999)
    assert (n2, mn2, mx2) == (5_000, 20_000, 24_999)

    # global sanity: all numbers present exactly once in [0..24_999]
    all_numbers = (
        pl.scan_parquet(root / f'*/{glob_exp}')
        .select('number')
        .collect()
        .to_series()
        .sort()
        .to_list()
    )
    assert len(all_numbers) == 25_000
    assert all_numbers == list(range(25_000))

