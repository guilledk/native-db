from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from native_db.table.builder import TableBuilder

from native_db._testing import (
    blockchain_ctx,
    block_stream,
)


def test_definitions(tmp_path: Path):
    ctx = blockchain_ctx(tmp_path)
    print(ctx.blocks.pretty_str())
    struct_cls = ctx.blocks.struct
    assert isinstance(
        struct_cls(
            number=0, timestamp=datetime.now(tz=timezone.utc), hash='test'
        ),
        ctx.blocks.struct,
    )
