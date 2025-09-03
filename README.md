# native-db

*Status: work in progress*

An attempt at a timeseries database (with some key-value features) using only dataframes and `linux`.

Modern columnar formats and libraries make on‑disk querying fast and expressive. Writing is harder: the best settings depend on the actual schema and access patterns, and most dataframe libraries expect full‑frame writes rather than true append. `native-db` focuses on the write path for large, append‑heavy workloads while keeping reads as standard `polars` scans.

## Features

`native-db` provides a set of focused extensions around Polars:

- **Extended data types**
  - `Keyword` for short strings that benefit from dictionary encoding.
  - `Mono` for monotonic integer identifiers (e.g., block numbers, IDs).

- **Schema with usage hints (`Table`)**
  - A thin wrapper over a Polars schema that adds optionality, uniqueness, sort order, and average value size hints (for dynamic types).  
    These hints help size Parquet row groups, decide dictionary use, and keep the write path predictable.

- **Incremental builders (`TableBuilder`)**
  - Append rows as Python iterables.  
    Unique columns can be guarded by an on‑disk bloom filter, dropping duplicates early.

- **Streaming writer (`TableWriter`)**
  - Stage/commit write cycle for datasets larger than memory.  
    Respects sort/uniqueness hints, handles partitioning, and rolls files to a configured target size.

- **Dataset context and cached transforms (`Context`, `Transform`)**
  - Group multiple tables, some local and some remote. Declare derived views as `Transform`s; the first materialization is cached to disk and later served via `scan_parquet`.

## Example

Block and transaction dataset declaration:

```python
# start by declaring a context with two tables
ctx = Context(
    datadir='/data/testnet',
    tables=(
        # block time series
        Table(
            'block',  # in-code table name
            'static/block',  # on-disk location relative to datadir
            (
                # 32-bit monotonic ascending block number
                ('number', Mono(size=4), TypeHints(sort='asc')),
                # microsecond UTC block timestamp
                ('timestamp', pl.Datetime(time_unit='us', time_zone='UTC')),
                # 256-bit hash as hex string
                ('hash', Keyword, TypeHints(avg_str_size=64)),
            ),
            # specify on-disk format: csv, ipc or parquet
            format='parquet',
            # compression settings (depends on format)
            compression='zstd',
            # partition on the block number column, each bucket dir will contain
            # 10 million rows, each with a unique block number
            partitioning=MonoPartition(on_column='number', row_size=10_000_000),
            writer_opts=TableWriterOptions(
                # accumulate a million rows in staging before committing
                commit_threshold=1_000_000,
                # exact size of staged frames
                stage_threshold=1_000_000,
                # final file max size
                rows_per_file=1_000_000,
            ),
        ),
        # transaction dictionary
        Table(
            'transaction',
            'static/transaction',
            (
                # 256-bit hash as hex string
                ('hash', Keyword, TypeHints(avg_str_size=64)),
                # microsecond UTC tx timestamp
                ('timestamp', pl.Datetime(time_unit='us', time_zone='UTC')),
                # tx block number
                ('block_number', Mono(size=4), TypeHints(sort='asc')),
                # index of tx in block
                ('block_index', Mono(size=4), TypeHints(sort='asc')),
                # 160-bit Ethereum-style addresses in hex
                ('from', Keyword, TypeHints(avg_str_size=40)),
                ('to', Keyword, TypeHints(avg_str_size=40)),
                # 256-bit unsigned int (as bytes)
                ('value', pl.Binary, TypeHints(avg_str_size=16)),
                # optional contract address
                ('contract_addr', Keyword, TypeHints(avg_str_size=40, optional=True)),
                # optional input params
                ('params', pl.Binary, TypeHints(avg_str_size=256, optional=True)),
            ),
            format='parquet',
            compression='zstd',
            # partition on the first char of the hash column allowing faster
            # searches by hash
            partitioning=DictionaryPartition(on_column='hash', depth=1),
        ),
    )
)

# use builder to build table frames row by row
builder = ContextBuilder(ctx)

# row stream type hints
BlockType = tuple[int, datetime, str]
TxType = tuple[str, datetime, int, int, str, str, bytes, str | None, bytes | None]
BlockStreamItem = tuple[BlockType, list[TxType]]

# consume a stream
for block_row, tx_rows in block_stream:
    builder.append('block', block_row)
    if tx_rows:
        builder.extend('transaction', tx_rows)

# flush all in-memory frames to the on-disk staging area
staged = await builder.stage(drain=True)

async with open_ctx_writer(ctx) as writer:
    # hand staged frames to the writer; it will commit them according to
    # partitioning and writer options from the table definition
    for table_name, frames in staged.items():
        for frame in frames:
            await writer.stage_direct(table_name, frame)

# getattr on ctx with a table name returns the Table object
lazy_blocks: pl.LazyFrame = ctx.block.scan()  # scan from the configured format
```

Reading ~22 million block rows yields a layout such as:

```
# partition every 10 million
/data/testnet/static/block
├── bucket=0
│   ├── 00000.parquet
│   ├── 00001.parquet
│   └── ...
├── bucket=1
│   ├── 00000.parquet
│   └── ...
└── bucket=2
    ├── 00000.parquet
    └── 00001.parquet

# one partition for each hex char
/data/testnet/static/transaction
├── char0=0
│   └── 00000.parquet
├── char0=1
│   ├── 00000.parquet
│   └── 00001.parquet
...
└── char0=f
    └── 00000.parquet
```

## Design

### Overview

The write path is optimized for **append‑heavy** ingestion that may arrive in batches and not always in perfect order. It has two cooperating pieces:

1. **Builders stage exact‑size frames**  
   `ContextBuilder` holds one `TableBuilder` per table. Each builder accumulates rows and only flushes a frame when it reaches the table’s `stage_threshold` (exact sizing). Frames are named `frame-{index}.{rows}-rows.{fmt}` and land under `table/.staging/`. Frames keep the caller’s index order per table.  
2. **Writers commit ordered batches**  
   `ContextWriter` feeds each table’s `TableWriter`. The writer reorders any out‑of‑order frames in memory, then commits them in order once a global `commit_threshold` (rows) is met-packing as many staged frames as needed per commit.

#### Stage cycle (builders)

- **Append**: user code pushes rows into `TableBuilder` per table. Bloom filters (if configured via `unique` hints) filter duplicates on the hot path.  
- **Flush**: when a builder reaches `stage_threshold` rows it emits a **single exact‑size** `DataFrame`. The `ContextBuilder` writes it as `IPC/Parquet` to `.staging/` and records the frame index.  
- **Drain**: at end‑of‑stream, builders optionally emit one final under‑filled remainder.

#### Commit cycle (writers)

- **Ordering**: `TableWriter` buffers staged frames by index and advances a "next expected" pointer; gaps hold later frames until missing ones arrive.  
- **Thresholding**: once total staged rows reach `commit_threshold`, the writer creates a commit directory `.staging/.commit-{n}` and moves the participating frames there.
- **Partition & sort**: a single lazy concat of commit frames is prepared, partition keys are derived (see partitioners below), and per‑partition sort is applied when hints request it.
- **File rolling**: produced parts are integrated into the final table location. Parts are concatenated up to `rows_per_file`, splitting when needed to keep files uniform.
- **Atomicity**: work happens in the commit directory and then moves/renames into place. Only one writer is allowed per table path.  
- **Finalization**: on shutdown or `drain()`, any remaining frames are committed in a last batch.

### Partitioners

Partitioners derive **hive‑style** directory keys from a chosen column. They do not drop or rename user columns; they add derived key columns used by the sink.

- **`MonoPartition(on_column, row_size)`**  
  For monotonic integers. Buckets are computed as `value // row_size`, producing paths like `bucket=0/`, `bucket=1/`, etc. Good for block numbers or IDs.

- **`TimePartition(on_column, kind)`**  
  For temporal columns. Keys can be `year`, `year-month`, or `year-month-day`, yielding paths like `year=2024/month=09/`.

- **`DictionaryPartition(on_column, depth, pad_to=None, lowercase=False, signed=False)`**  
  For string or numeric IDs. Takes the left‑to‑right most‑significant characters of the **string representation** of the value, exposing `char0`, `char1`, … up to `depth`. This creates paths such as `char0=a/`, `char0=5/`. For numbers, set `pad_to` to make bucket prefixes stable.

- **`RowLenPartition(on_column, row_size)`**  
  Simpler "row count" bucketing for cases where keys are not meaningful.

### On‑disk layout

Each partition value houses **rolled parts** using a zero‑padded sequential filename, e.g. `00000.parquet`, `00001.parquet`, up to the configured `rows_per_file`. Frames staged for a commit are never left behind; the commit directory is removed after integration.

## Notes

- The name `Table` here is a project type distinct from `pyarrow.Table` (TODO: diff name?).
- This repository is intentionally limited in scope: simple, predictable
  write‑time behaviors over a small set of patterns that are common in
  time‑series and key‑value workloads.
