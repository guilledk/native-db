from __future__ import annotations

import io

from typing import Iterable, TYPE_CHECKING

import polars as pl
from polars._typing import IpcCompression, PythonDataType


if TYPE_CHECKING:
    from native_db.table import Table


class TableBuilder:
    '''
    Lightweight column store for building in-memory table bytes.

    - Append/extend accumulate Python values per column (no Arrow arrays involved).
    - flush() materializes a pl.DataFrame once and writes a *single* in‑memory
      stream, clearing internal buffers for reuse.

    '''

    def __init__(self, table: 'Table', target_row_size: int | None = None):
        self._table = table

        cols = table.schema.columns
        self._names = [c.name for c in cols]
        self._ncols = len(self._names)

        # one python list per column (schema order)
        self._col_lists: list[list[PythonDataType]] = [[] for _ in self._names]
        self._first_col = self._col_lists[0] if self._col_lists else []

        # cache sort options as column names (avoid expr construction on hot path)
        self._sort_names: list[str] = []
        self._sort_desc: list[bool] = []
        for c in cols:
            s = c.hints.sort
            if s == 'asc':
                self._sort_names.append(c.name)
                self._sort_desc.append(False)
            elif s == 'desc':
                self._sort_names.append(c.name)
                self._sort_desc.append(True)

        # cache non-nullable column names for a single-pass check at flush time
        self._nonnull_names = [c.name for c in cols if not c.hints.optional]

        # if set, flush must emit exactly this many rows
        self._target_rows = target_row_size

    def append(self, row: Iterable[PythonDataType]) -> None:
        cols = self._col_lists
        i = 0
        for v in row:
            cols[i].append(v)
            i += 1
        # no validation here (max speed)

    def extend(self, rows: Iterable[Iterable[PythonDataType]]) -> None:
        cols = self._col_lists
        for row in rows:
            i = 0
            for v in row:
                cols[i].append(v)
                i += 1
        # no validation here (max speed)

    def rows(self) -> int:
        return len(self._first_col) if self._col_lists else 0

    def flush(
        self, *, compression: IpcCompression = 'uncompressed'
    ) -> memoryview:
        '''
        Convert accumulated rows into a single in-memory IPC stream and clear/advance buffers.

        Validations performed here (only here):
          - column length consistency (or sufficient rows when target_row_size is set)
          - non-nullability (for produced slice only)
          - exact target_row_size semantics (raise if too few; keep remainder if too many)

        Returns:
          memoryview over the bytes of the IPC-formatted table.
        '''
        # quick exit
        if not self._col_lists or not self._first_col:
            return memoryview(b'')

        # compute per-column lengths once
        lengths = [len(col) for col in self._col_lists]
        min_len = min(lengths)
        max_len = max(lengths)

        # determine how many rows we will actually materialize
        target = self._target_rows

        if target is None:
            # enforce strict consistency if no target was requested
            if min_len != max_len:
                raise ValueError(
                    f'Column length mismatch at flush: min={min_len}, max={max_len}'
                )
            n_out = min_len
            if n_out == 0:
                return memoryview(b'')
        else:
            # must have at least target rows in every column
            if min_len < target:
                raise ValueError(
                    f'Not enough rows to flush target_row_size={target}: have min_col_len={min_len}'
                )
            n_out = target

        # build the data slice used to create the DataFrame
        if n_out == min_len == max_len:
            # fast path: whole buffer
            data = {
                name: col
                for name, col in zip(self._names, self._col_lists, strict=True)
            }
            consume_all = (
                target is None
            )  # if no target, we consume entire buffers
        else:
            # partial slice (first n_out rows)
            data = {
                name: col[:n_out]
                for name, col in zip(self._names, self._col_lists, strict=True)
            }
            consume_all = False

        df = pl.DataFrame(data, schema=self._table.schema.as_polars())

        # validate non-nullability (only for the emitted slice)
        if self._nonnull_names:
            checks = [
                pl.col(n).is_null().any().alias(n) for n in self._nonnull_names
            ]
            null_any = df.select(checks).row(0)
            for name, has_null in zip(
                self._nonnull_names, null_any, strict=True
            ):
                if has_null:
                    raise ValueError(
                        f'Tried to flush non-nullable column {name} with null values'
                    )

        # maybe sort (only the emitted slice)
        if self._sort_names:
            df = df.sort(by=self._sort_names, descending=self._sort_desc)

        # write IPC to an in-memory sink
        sink = io.BytesIO()
        df.write_ipc(sink, compression=compression)

        # advance/clear buffers:
        if consume_all:
            # keep capacity
            for col in self._col_lists:
                col.clear()
        else:
            # drop only the emitted prefix; keep remainder for next flush
            # (O(n) left pop per column—simple and predictable)
            n = n_out
            for col in self._col_lists:
                del col[:n]

        return sink.getbuffer()
