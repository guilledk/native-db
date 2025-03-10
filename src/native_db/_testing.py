from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Generator, Literal

import polars as pl

from native_db._utils import epoch
from native_db.dtypes import Keyword, Mono32, TypeHints
from native_db.table import Table


block_table = Table(
    'block',
    'static/blocks',
    (
        ('number', Mono32(allow_gaps=False), TypeHints(sort='asc')),
        ('timestamp', pl.Datetime(time_unit='us', time_zone='UTC')),
        ('hash', Keyword, TypeHints(avg_str_size=64))
    ),
    datadir=Path(__file__).parent.parent / 'tests/.test-db'
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
            hex_i + h[len(hex_i):]
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
                window = list(range(start, min(start + max_reorder_window, len(batches))))
                rnd.shuffle(window)
                emit_order.extend(window)
        else:
            emit_order = list(range(len(batches)))
            rnd.shuffle(emit_order)

    # 3. Yield
    for i in emit_order:
        yield batches[i]
