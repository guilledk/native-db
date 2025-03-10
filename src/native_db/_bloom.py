'''
# Overview

Multi reader/writer on-disk Bloom filter backed by Arrow IPC (Feather).

# Notes
    - Per-block lock file guards concurrent writers (POSIX flock).
    - Reads are lock-free and use a small mtime/size-validated block cache.
    - Each block file stores a single row: {block_id: UInt32, bits: Binary}

# Layout

    root/
        metadata.json
        blocks/
        block_id=000000/part-0.ipc
        block_id=000001/part-0.ipc
        ...

'''
from __future__ import annotations

from contextlib import contextmanager, suppress
import os
import math
import fcntl
from collections import OrderedDict
from pathlib import Path
import threading
from typing import Iterable, Sequence

import msgspec
import polars as pl
import xxhash


def _hash64(data: bytes, *, seed: int) -> int:
    '''
    Fast bytes -> int hashing function.

    '''
    return xxhash.xxh64_intdigest(data, seed=seed)


def _ensure_bytes(key: bytes | str) -> bytes:
    '''
    Collapse ambiguous string like into bytes.

    '''
    return key if isinstance(key, bytes) else key.encode('utf-8')


class BloomMeta(msgspec.Struct, frozen=True):
    '''
    Metadata for an on-disk blocked Bloom filter.
    Stored as metadata.json at the dataset root.

    '''

    B: int  # number of blocks
    block_bits: int  # bits per block (power of two)
    k: int  # number of hash functions
    seeds: tuple[int, int]  # seeds for double hashing (h1/h2)
    version: str = '1.0'  # version for schema evolution

    # (optional) documentation/deriveds
    n_capacity: int | None = None
    p_target: float | None = None
    m_total_bits: int | None = None


def bloom_params(n: int, p: float) -> tuple[int, int]:
    '''
    Given expected distinct items n and false-positive rate p, return (m bits, k hashes).

    '''
    if n <= 0 or not (0 < p < 1):
        raise ValueError('n must be > 0 and 0 < p < 1')
    m = math.ceil(-(n * math.log(p)) / (math.log(2) ** 2))
    k = max(1, round((m / n) * math.log(2)))
    return m, k


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class DiskBloom:

    def __init__(
        self,
        root: str | Path,
        *,
        # when creating a new dataset (metadata.json missing):
        N: int | None = None,
        P: float | None = None,
        block_bits: int = 1 << 20,  # 128 KiB
        seeds: tuple[int, int] = (0x12345678, 0x9ABCDEF0),
        # tuning:
        cache_blocks: int = 64,  # number of block bitmaps to cache for reads
    ) -> None:
        self.root: Path = Path(root)
        self.blocks_dir: Path = self.root / 'blocks'
        self.meta_path: Path = self.root / 'metadata.json'

        self.root.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)

        if self.meta_path.is_file():
            self.meta: BloomMeta = msgspec.json.decode(
                self.meta_path.read_bytes(), type=BloomMeta
            )
        else:
            # dataset bootstrap requires N and P
            if N is None or P is None:
                raise FileNotFoundError(
                    'metadata.json not found; pass N (capacity) and P (target FPR) to create it'
                )
            if not _is_power_of_two(block_bits):
                raise ValueError('block_bits must be a power of two')

            m, k = bloom_params(N, P)
            B = math.ceil(m / block_bits)

            self.meta = BloomMeta(
                B=B,
                block_bits=block_bits,
                k=k,
                seeds=seeds,
                version='1.1',
                n_capacity=N,
                p_target=P,
                m_total_bits=m,
            )
            self._write_meta()

        # cached constants
        self._mask: int = self.meta.block_bits - 1
        self._ext: str = 'ipc'
        self._k: int = self.meta.k
        self._B: int = self.meta.B

        # tiny LRU cache: bid -> (mtime_ns, size, bytearray)
        self._cache_cap = max(0, int(cache_blocks))
        self._cache: 'OrderedDict[int, tuple[int, int, bytearray]]' = (
            OrderedDict()
        )
        self._cache_lock = threading.Lock()

    def _write_meta(self) -> None:
        self.meta_path.write_bytes(msgspec.json.encode(self.meta))

    def _block_dir(self, bid: int) -> Path:
        return self.blocks_dir / f'block_id={bid:06d}'

    def _block_path(self, bid: int) -> Path:
        return self._block_dir(bid) / f'part-0.{self._ext}'

    def _lock_path(self, bid: int) -> Path:
        return self._block_dir(bid) / '.lock'

    def _ensure_block_dir(self, bid: int) -> None:
        self._block_dir(bid).mkdir(parents=True, exist_ok=True)

    def _h1_h2(self, key: bytes | str) -> tuple[int, int]:
        b = _ensure_bytes(key)
        s1, s2 = self.meta.seeds
        return _hash64(b, seed=s1), _hash64(b, seed=s2)

    def _empty_bits(self) -> bytearray:
        return bytearray(self.meta.block_bits // 8)

    def _read_bits_from_disk(self, bid: int) -> tuple[int, int, bytearray]:
        '''
        Read the block file (single-row IPC/Feather). If missing, return zeroed bytes.
        Returns (mtime_ns, size, bits).
        '''
        path = self._block_path(bid)
        if not path.is_file():
            return (0, 0, self._empty_bits())

        st = path.stat()
        # read only the 'bits' column to minimize work
        df = pl.read_ipc(path, memory_map=True, columns=['bits'])
        if df.height == 0:
            return (st.st_mtime_ns, st.st_size, self._empty_bits())

        # avoid to_list() -> use direct scalar extraction
        bits = df.get_column('bits').item(0)  # type: ignore[assignment]
        # ensure mutable for in-place bit checks (indexing is faster on bytearray)
        return (st.st_mtime_ns, st.st_size, bytearray(bits))

    def _save_bits(self, bid: int, bits: bytes | bytearray) -> None:
        '''
        Atomically write the block file (single-row IPC/Feather, uncompressed).
        Also refresh read-cache coherently.
        '''
        self._ensure_block_dir(bid)
        path = self._block_path(bid)
        tmp = path.with_suffix(path.suffix + '.tmp')

        df = pl.DataFrame(
            {'block_id': [bid], 'bits': [bytes(bits)]},
            schema=pl.Schema({'block_id': pl.UInt32, 'bits': pl.Binary}),
        )
        df.write_ipc(tmp, compression='uncompressed')
        os.replace(tmp, path)  # atomic on POSIX

        # refresh cache entry (keep this block hot for readers)
        if self._cache_cap:
            st = path.stat()
            with self._cache_lock:
                self._cache[bid] = (st.st_mtime_ns, st.st_size, bytearray(bits))
                # another thread could evict between set and move; be tolerant
                try:
                    self._cache.move_to_end(bid, last=True)
                except KeyError:
                    pass
                while len(self._cache) > self._cache_cap:
                    self._cache.popitem(last=False)

    def _get_bits_readonly(self, bid: int) -> bytearray:
        '''
        Return current bits for 'bid', using a tiny LRU validated by (mtime_ns, size).
        Cache entries are refreshed automatically after writes.
        '''
        path = self._block_path(bid)
        if self._cache_cap:
            with self._cache_lock:
                entry = self._cache.get(bid)

            if entry is not None:
                mtime_ns, size, bits = entry
                try:
                    st = path.stat()

                except FileNotFoundError:
                    st = None

                if (st is None and mtime_ns == 0 and size == 0) or (
                    st is not None
                    and st.st_mtime_ns == mtime_ns
                    and st.st_size == size
                ):
                    # cache is still valid
                    with self._cache_lock:
                        # tolerate concurrent eviction
                        try:
                            self._cache.move_to_end(bid, last=True)
                        except KeyError:
                            pass

                    return bits

        # (cache miss) -> read from disk and insert
        mtime_ns, size, bits = self._read_bits_from_disk(bid)
        if self._cache_cap:
            with self._cache_lock:
                self._cache[bid] = (mtime_ns, size, bits)
                try:
                    self._cache.move_to_end(bid, last=True)
                except KeyError:
                    pass
                while len(self._cache) > self._cache_cap:
                    self._cache.popitem(last=False)
        return bits

    def might_contain(self, key: bytes | str) -> bool:
        '''
        Check membership; may return false positives but not false negatives.
        '''
        h1, h2 = self._h1_h2(key)
        bid = h1 % self._B
        bits = self._get_bits_readonly(bid)

        mask = self._mask
        k = self._k
        for i in range(k):
            bit = (h1 + i * h2) & mask
            byte = bit >> 3
            off = bit & 7
            if (bits[byte] & (1 << off)) == 0:
                return False
        return True

    def might_contain_many(self, keys: Sequence[bytes | str]) -> list[bool]:
        '''
        Vectorized membership: group by block, read each block once, answer in input order.
        '''
        n = len(keys)
        if n == 0:
            return []
        # bucketize indices by block id
        buckets: dict[int, list[int]] = {}
        h_pairs: list[tuple[int, int]] = [None] * n  # type: ignore
        for idx, key in enumerate(keys):
            h1, h2 = self._h1_h2(key)
            h_pairs[idx] = (h1, h2)
            bid = h1 % self._B
            buckets.setdefault(bid, []).append(idx)

        out = [False] * n
        mask = self._mask
        k = self._k

        for bid, idxs in buckets.items():
            bits = self._get_bits_readonly(bid)
            for idx in idxs:
                h1, h2 = h_pairs[idx]
                ok = True
                # unrolled-ish small loop
                for i in range(k):
                    bit = (h1 + i * h2) & mask
                    b = bits[bit >> 3]
                    if (b & (1 << (bit & 7))) == 0:
                        ok = False
                        break
                out[idx] = ok

        return out

    def add(self, key: bytes | str) -> None:
        '''
        Idempotent add: sets k bits in the block file that 'key' maps to.
        '''
        h1, h2 = self._h1_h2(key)
        bid = h1 % self._B
        with self._with_block_lock(bid):
            # Writers must start from authoritative bytes on disk to avoid
            # lost updates when cache is stale under concurrency.
            bits = self._read_bits_from_disk(bid)[2][:]  # copy for RMW
            mask = self._mask
            k = self._k
            for i in range(k):
                bit = (h1 + i * h2) & mask
                byte = bit >> 3
                off = bit & 7
                bits[byte] |= 1 << off
            self._save_bits(bid, bits)

    def add_many(self, keys: Iterable[bytes | str]) -> None:
        '''
        Efficient batched adds: groups keys by block id and writes each block once.
        '''
        # bucketize keys by block id
        buckets: dict[int, list[tuple[int, int]]] = {}
        for key in keys:
            h1, h2 = self._h1_h2(key)
            bid = h1 % self._B
            buckets.setdefault(bid, []).append((h1, h2))

        # per-block read-modify-write under lock
        mask = self._mask
        k = self._k
        for bid, hh in buckets.items():
            with self._with_block_lock(bid):
                bits = self._read_bits_from_disk(bid)[2][:]  # copy for RMW
                for h1, h2 in hh:
                    for i in range(k):
                        bit = (h1 + i * h2) & mask
                        byte = bit >> 3
                        off = bit & 7
                        bits[byte] |= 1 << off
                self._save_bits(bid, bits)

    @contextmanager
    def _with_block_lock(self, bid: int):
        '''
        POSIX-only exclusive flock on a per-block .lock file.

        '''
        self._ensure_block_dir(bid)
        lock_path = self._lock_path(bid)
        fh = open(lock_path, 'a+')
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
        with suppress(Exception):
            fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    @property
    def capacity(self) -> int | None:
        return self.meta.n_capacity

    @property
    def target_fpr(self) -> float | None:
        return self.meta.p_target

    @property
    def total_bits(self) -> int | None:
        return self.meta.m_total_bits
