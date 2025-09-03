
import fnmatch
import logging
import os
from pathlib import Path
from typing import IO, Callable, Literal, Protocol, Sequence

import polars as pl
from polars._typing import PartitioningScheme

import pyarrow.parquet as pq

from native_db.lowlevel import PolarsExecutor


log = logging.getLogger(__name__)


FrameFormats = Literal['csv', 'ipc', 'parquet']


def format_from_path(path: str | Path) -> FrameFormats:
    '''
    Given a file path figure out its format from the suffix.

    '''
    path = Path(path)
    splt = path.name.split('.')

    if splt[-1] == 'tmp':
        splt.pop()

    match splt[-1]:
        case 'csv' | 'ipc' | 'parquet':
            return splt[-1]

        case _:
            raise ValueError(
                'Format not specified and target file has unknown suffix:'
                f' {path.suffix}'
            )


class ScanFrameFn(Protocol):
    '''
    Minimun shared call signature between supported polars scan_* functions

    '''
    def __call__(
        self,
        source: str | Path | IO[bytes] | bytes | list[str] | list[Path] | list[IO[bytes]] | list[bytes],
        *,
        rechunk: bool,
    ) -> pl.LazyFrame: ...


def scan_frame(
    path: str | Path,
    *,
    format: FrameFormats | None = None,
    rechunk: bool = False,
    glob: bool = False,
    **kwargs
) -> pl.LazyFrame:
    '''
    Scan any supported frame file into a LazyFrame.

    '''
    if not format:
        format = format_from_path(path)

    scan_fn: ScanFrameFn
    match format:
        case 'csv':
            scan_fn = pl.scan_csv

        case 'ipc':
            scan_fn = pl.scan_ipc

        case 'parquet':
            scan_fn = pl.scan_parquet
            kwargs['glob'] = glob

    return scan_fn(path, rechunk=rechunk, **kwargs)


def sink_frame(
    frame: pl.LazyFrame,
    path: str | Path | PartitioningScheme,
    *,
    format: FrameFormats | None = None,
    **kwargs
) -> pl.LazyFrame:
    '''
    Get a lazy sink-to-path operation for any supported frame format.

    '''
    if not format:
        if isinstance(path, str | Path):
            format = format_from_path(path)

        else:
            raise ValueError(
                'sink_frame requires format argument if sinking a partitioned frame'
            )

    sink_fn: Callable
    match format:
        case 'csv':
            sink_fn = frame.sink_csv

        case 'ipc':
            sink_fn = frame.sink_ipc

        case 'parquet':
            sink_fn = frame.sink_parquet

    if 'lazy' not in kwargs:
        kwargs['lazy'] = True

    if 'mkdir' not in kwargs:
        kwargs['mkdir'] = True

    return sink_fn(path, **kwargs)


def parquet_row_len(path: str | Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


async def row_lens(
    paths: Sequence[str | Path],
    executor: PolarsExecutor | None = None
) -> tuple[int, ...]:
    '''
    Get row length of multiple on-disk frames in parallel.

    '''
    all_parquet = not any(Path(p).suffix != 'parquet' for p in paths)
    if all_parquet:
        return tuple((
            parquet_row_len(p)
            for p in paths
        ))

    frames = tuple((
        scan_frame(path).select(pl.len())
        for path in paths
    ))

    dfs: list[pl.DataFrame]
    if executor:
        dfs = await executor.collect_all(frames)

    else:
        dfs = await pl.collect_all_async(frames)

    return tuple((
        df.item()
        for df in dfs
    ))


async def row_len(
    path: Path,
    executor: PolarsExecutor | None = None
) -> int:
    '''
    Get row length of an on-disk frame.

    '''
    if path.suffix == 'parquet':
        return parquet_row_len(path)

    frame = (
        scan_frame(path)
        .select(pl.len())
    )

    df: pl.DataFrame
    if executor:
        df = await executor.collect(frame)

    else:
        df = await frame.collect_async()

    return df.item()


async def rewrite_frame(
    frame: pl.LazyFrame,
    location: Path,
    *,
    executor: PolarsExecutor | None = None,
    **kwargs
) -> None:
    '''
    In order to atomically overwrite an on disk-frame, write full modified
    frame to a temporal location adjacent to target, then replace the target 
    with the new temporal frame.

    '''
    tmp = location.with_suffix(location.suffix + ".tmp")

    res = sink_frame(
        frame,
        tmp,
        **kwargs
    )

    if executor:
        await executor.collect(res)

    tmp.replace(location)


async def concat_or_split_frame(
    source: Path,
    target: Path,
    split: Path,
    max_rows: int,
    *,
    source_len: int | None = None,
    target_len: int | None = None,
    executor: PolarsExecutor | None = None
) -> bool:
    '''
    Concatenate rows from source into target, spliting into a new file if
    `max_rows` is reached on target.

    Return true if split location was used.

    '''
    # figure out row counts if not provided
    if not source_len and not target_len:
        source_len, target_len = await row_lens((source, target), executor=executor)

    elif source_len is None:
        source_len = await row_len(source, executor=executor)

    elif target_len is None:
        target_len = await row_len(target, executor=executor)

    # if target is empty and source row count <= max_rows just place source
    # in target location
    if target_len == 0 and source_len <= max_rows:
        source.replace(target)
        return False

    # depending on remaining space on target frame we perform diferent disk ops
    space = max_rows - target_len

    if space == 0:
        # no space remaining, just place source frame into split location
        source.replace(split)
        return True

    if source_len <= space:
        # target + source rows can fit directly concatenated
        merged = pl.concat((scan_frame(target), scan_frame(source)), rechunk=False)
        await rewrite_frame(merged, target, executor=executor)
        return False

    if space < 0:
        # target frame is over max_rows, just write source to split location
        source.replace(split)
        return True

    # some space is left on target, but not enough to directly concat

    source_frame = scan_frame(source)

    # create filled target frame concatenating up to `space` rows from
    # source
    fill = source_frame.slice(0, space)
    merged = pl.concat((scan_frame(target), fill), rechunk=False)
    await rewrite_frame(merged, target, executor=executor)

    # write the remainder to the split location
    remainder = source_frame.slice(space, None)
    await rewrite_frame(remainder, split, executor=executor)
    return True



def _part_index(path: Path) -> int:
    return int(path.name.split('-')[-1].split('.')[0])

def find_last_part(bucket_path: Path, pattern: str) -> tuple[Path | None, int]:
    max_idx = -1
    last = None
    for entry in os.scandir(bucket_path):
        if entry.is_file() and fnmatch.fnmatch(entry.name, pattern):
            idx = _part_index(Path(entry.name))
            if idx > max_idx:
                max_idx = idx
                last = Path(entry.path)
    return last, max_idx
