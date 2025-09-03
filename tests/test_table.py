from io import BytesIO
from datetime import datetime, timezone

import polars as pl

from native_db.table.builder import TableBuilder

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

    df = builder.flush_frame()
    iosink = BytesIO()
    df.write_ipc(iosink)

    frame = pl.scan_ipc(iosink).collect()

    assert frame.select(pl.len()).item() == len(rows)
    assert frame.schema == block_table.schema.as_polars()
    assert frame.rows() == rows
