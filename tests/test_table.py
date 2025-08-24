from io import BytesIO
from datetime import datetime, timezone

import pytest
import polars as pl

from native_db._utils import NativeDBWarning
from native_db.table.builder import TableBuilder
from native_db.table.writer import TableWriterOptions

from native_db._testing import (
    block_table,
    block_stream,
)


def test_definitions():
    print(block_table.pretty_str())
    struct_cls = block_table.struct
    assert isinstance(
        struct_cls(
            number=0, timestamp=datetime.now(tz=timezone.utc), hash='test'
        ),
        block_table.struct,
    )


def test_builder():
    builder = TableBuilder(block_table)

    rows = [row for row in block_stream()]
    builder.extend(rows)

    frame = pl.scan_ipc(BytesIO(builder.flush())).collect()

    assert frame.select(pl.len()).item() == len(rows)
    assert frame.schema == block_table.schema.as_polars()
    assert frame.rows() == rows


def test_writer_options_warn_on_misaligned_threshold():
    with pytest.warns(NativeDBWarning):
        _ = TableWriterOptions(commit_threshold=1500, rows_per_file=1000)
