from __future__ import annotations

import os
from typing import Iterable, TYPE_CHECKING

import polars as pl
from polars._typing import PythonDataType

from native_db.lowlevel.bloom import DiskBloom


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
        self._unique_idx: list[int] = []
        self._blooms: dict[int, DiskBloom] = {}
        for i, c in enumerate(cols):
            s = c.hints.sort
            if s == 'asc':
                self._sort_names.append(c.name)
                self._sort_desc.append(False)
            elif s == 'desc':
                self._sort_names.append(c.name)
                self._sort_desc.append(True)

            if c.hints.unique:
                self._unique_idx.append(i)
                bpath = table.local_path / '.staging' / f'.{c.name}-{os.getpid()}.bloom'
                bpath.parent.mkdir(parents=True, exist_ok=True)
                self._blooms[i] = DiskBloom(bpath, N=30_000_000, P=0.001)

        # cache non-nullable column names for a single-pass check at flush time
        self._nonnull_names = [c.name for c in cols if not c.hints.optional]

        # if set, flush must emit exactly this many rows
        self._target_rows = target_row_size

    def append(self, row: Iterable[PythonDataType]) -> None:
        cols = self._col_lists
        if not self._blooms:
            i = 0
            for v in row:
                cols[i].append(v)
                i += 1
            # no validation here (max speed)
            return

        row = tuple(row)

        bloom_add = []
        for bi in self._unique_idx:
            v = row[bi]
            if self._blooms[bi].might_contain(v):
                return

            bloom_add.append((self._blooms[bi], v))

        i = 0
        for v in row:
            cols[i].append(v)
            i += 1

        if bloom_add:
            for bloom, val in bloom_add:
                bloom.add(val)


    def extend(self, rows: Iterable[Iterable[PythonDataType]]) -> None:
        cols = self._col_lists
        if not self._blooms:
            # Fast path: just append, once.
            for row in rows:
                for i, v in enumerate(row):
                    cols[i].append(v)
            return  # avoid duplicate work below

        # Bloom path: drop only offending rows; keep good ones.
        appended_bloom_vals: list[tuple[int, PythonDataType]] = []
        for _row in rows:
            row = tuple(_row)
            # Check all unique fields for this row
            for bi in self._unique_idx:
                if self._blooms[bi].might_contain(row[bi]):
                    # duplicate on a unique column: skip just this row
                    break
            else:
                # Append full row
                for i, v in enumerate(row):
                    cols[i].append(v)
                # Record bloom additions (defer writes until row is accepted)
                for bi in self._unique_idx:
                    appended_bloom_vals.append((bi, row[bi]))

        # Update blooms only for rows we actually appended
        for bi, val in appended_bloom_vals:
            self._blooms[bi].add(val)

    def rows(self) -> int:
        return len(self._first_col) if self._col_lists else 0

    def flush_frame(self, *, allow_underfilled: bool = False, drain: bool = False) -> pl.DataFrame:
        """
        - Normal mode (target_row_size unset): same behavior as today.
        - target_row_size set & underfilled:
            * if drain=True and we have >0 rows -> emit the remainder now
            * elif allow_underfilled=True -> return EMPTY (do not advance buffers)
            * else -> raise (preserve strict behavior)
        """
        if not self._col_lists or not self._first_col:
            return self._table.empty().collect()

        lengths = [len(col) for col in self._col_lists]
        min_len = min(lengths); max_len = max(lengths)
        target = self._target_rows

        if target is None:
            if min_len != max_len:
                raise ValueError(f'Column length mismatch at flush: min={min_len}, max={max_len}')
            n_out = min_len
            if n_out == 0:
                return self._table.empty().collect()
        else:
            if min_len < target:
                if drain and min_len > 0:
                    n_out = min_len                # final remainder
                elif allow_underfilled:
                    return self._table.empty().collect()  # stay put (no-op)
                else:
                    raise ValueError(
                        f'Not enough rows to flush target_row_size={target}: have min_col_len={min_len}'
                    )
            else:
                n_out = target

        if n_out == min_len == max_len:
            data = {name: col for name, col in zip(self._names, self._col_lists, strict=True)}
            consume_all = (target is None)
        else:
            data = {name: col[:n_out] for name, col in zip(self._names, self._col_lists, strict=True)}
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

        if consume_all:
            for col in self._col_lists: col.clear()
        else:
            for col in self._col_lists: del col[:n_out]

        if self._table.partitioning:
            df = self._table.partitioning.prepare(df)

        return df

