import os
import random
import threading
import multiprocessing as mp
from pathlib import Path

import pytest
import polars as pl
import xxhash

from native_db._bloom import BloomMeta, DiskBloom, bloom_params


def test_bootstrap_and_metadata_math(tmp_path: Path):
    # Small block to force multiple blocks and exercise math
    N, P = 1_000, 0.01
    block_bits = 1 << 10  # 1024 bits
    seeds = (0x11111111, 0x22222222)

    db = DiskBloom(
        tmp_path,
        N=N,
        P=P,
        block_bits=block_bits,
        seeds=seeds,
    )

    # Metadata persisted on disk
    meta_path = tmp_path / 'metadata.json'
    assert meta_path.is_file()

    # Check meta mirrors inputs/derived values
    m_expected, k_expected = bloom_params(N, P)
    assert isinstance(db.meta, BloomMeta)
    assert db.meta.k == k_expected
    assert db.meta.m_total_bits == m_expected
    assert db.meta.n_capacity == N
    assert db.meta.p_target == P
    assert db.meta.block_bits == block_bits
    assert db.meta.seeds == seeds

    # Number of blocks is ceil(m / block_bits)
    import math

    assert db.meta.B == math.ceil(m_expected / block_bits)

    # Internals derived from format choice (file extension)
    # (_ext is private, but we can assert by writing one block later)


def test_persistence_across_instances(tmp_path: Path):
    key = 'hello-world'
    db1 = DiskBloom(tmp_path, N=100, P=0.01, block_bits=1 << 12)
    assert not db1.might_contain(
        key
    )  # may be FP, but almost surely False first
    db1.add(key)  # sets bits in one block

    # Re-open from existing metadata.json (no N/P required)
    db2 = DiskBloom(tmp_path)
    assert db2.might_contain(key)  # must not be a false negative


def test_add_and_membership_no_false_negatives(tmp_path: Path):
    # Use modest capacity to keep FP rate reasonable but focus on "no false negatives"
    N, P = 5_000, 0.01
    db = DiskBloom(tmp_path, N=N, P=P, block_bits=1 << 15)

    keys = [f'k{i}' for i in range(2000)]
    for k in keys:
        db.add(k)

    # Must be True for all inserted keys (Bloom property)
    for k in keys:
        assert db.might_contain(k)


def test_add_many_batches_across_blocks(tmp_path: Path):
    # Force many small blocks to exercise grouping/bucketing logic
    db = DiskBloom(tmp_path, N=10_000, P=0.01, block_bits=1 << 10)

    keys = [f'user:{i}' for i in range(5000)]
    db.add_many(keys)  # grouped per block then RMW once

    # All inserted keys must be reported as present (no false negatives)
    for k in keys[:200]:  # sample (keep test fast)
        assert db.might_contain(k)

    # Layout: at least one block directory exists with a single file named "part-0.<ext>"
    blocks_dir = tmp_path / 'blocks'
    block_dirs = [p for p in blocks_dir.glob('block_id=*') if p.is_dir()]
    assert block_dirs, 'no block directories created'
    some_block = block_dirs[0]
    files = list(some_block.glob(f'part-0.ipc'))
    assert files, f'expected part-0.ipc in {some_block}'


def test_basic_concurrency_best_effort_locking(tmp_path: Path):
    # The lock is POSIX-only best effort; verify threads don't corrupt state
    db = DiskBloom(tmp_path, N=50_000, P=0.01, block_bits=1 << 14)

    keys_a = [f'a{i}' for i in range(2000)]
    keys_b = [f'b{i}' for i in range(2000)]

    def add_many(keys):
        for k in keys:
            db.add(k)

    t1 = threading.Thread(target=add_many, args=(keys_a,))
    t2 = threading.Thread(target=add_many, args=(keys_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # No false negatives for inserted keys
    for k in keys_a[:200] + keys_b[:200]:
        assert db.might_contain(k)


def test_reopen_without_params_requires_existing_metadata(tmp_path: Path):
    # metadata.json missing -> must pass N and P (bootstrap)
    with pytest.raises(FileNotFoundError):
        _ = DiskBloom(tmp_path)  # no N/P and no metadata yet


def test_invalid_block_bits(tmp_path: Path):
    # block_bits must be a power of two
    with pytest.raises(ValueError):
        _ = DiskBloom(tmp_path, N=100, P=0.1, block_bits=1000)


def test_bloom_params_input_validation():
    with pytest.raises(ValueError):
        bloom_params(0, 0.01)  # n must be > 0
    with pytest.raises(ValueError):
        bloom_params(100, 1.5)  # 0 < p < 1


def test_accessors_and_meta_roundtrip(tmp_path: Path):
    N, P = 1234, 0.02
    db = DiskBloom(
        tmp_path,
        N=N,
        P=P,
        block_bits=1 << 13,
    )
    # Accessor helpers reflect metadata (may be None if missing, but here set)
    assert db.capacity == N
    assert db.target_fpr == P
    m, _ = bloom_params(N, P)
    assert db.total_bits == m

    # Rewrite metadata on disk then reload (sanity for _write_meta path)
    db._write_meta()  # internal, but harmless to exercise
    db2 = DiskBloom(tmp_path)
    assert db2.capacity == N
    assert db2.target_fpr == P
    assert db2.total_bits == m


# Helpers
def _force_keys_for_block(db: DiskBloom, bid: int, n: int) -> list[str]:
    '''
    Generate 'n' string keys that map to a desired block id for this bloom.
    Uses the same hash seeds as DiskBloom (double hashing) to target bid.
    '''
    keys = []
    s1, s2 = db.meta.seeds  # uses same seeds as impl
    i = 0
    while len(keys) < n:
        k = f'block{bid}-key{i}'
        h1 = xxhash.xxh64_intdigest(k.encode(), seed=s1)  # matches _hash64
        if (h1 % db.meta.B) == bid:
            keys.append(k)
        i += 1
    return keys


def test_threaded_same_block_contention(tmp_path: Path):
    '''
    Many threads hammer a *single* block: this stresses the per-block lock and
    read-modify-write path. We assert no false negatives for inserted keys and
    that the block file remains readable as a valid single-row table afterward.
    '''
    db = DiskBloom(
        tmp_path, N=100_000, P=0.01, block_bits=1 << 12
    )  # flock file locking per block
    # Choose a target block and craft keys for it
    target_bid = 0
    keys = _force_keys_for_block(db, target_bid, 5000)

    # Split among threads
    shards = [keys[i::8] for i in range(8)]

    def worker(part):
        for k in part:
            db.add(k)  # read-modify-write under lock

    threads = [threading.Thread(target=worker, args=(sh,)) for sh in shards]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All inserted keys must be present (no FNs by construction)
    for k in random.sample(keys, 200):
        assert db.might_contain(k)

    # Block file should be a valid single-row Parquet/IPC with 'bits' column
    block_path = db._block_path(target_bid)  # path format part-0.<ext>
    assert block_path.is_file()
    df = pl.read_ipc(block_path)
    assert df.height == 1 and 'bits' in df.columns


def test_threaded_cross_block_contention(tmp_path: Path):
    '''
    Threads add keys spread across many blocks to stress concurrent creation of
    multiple block dirs/files (ensuring mkdir + atomic replace are safe).
    '''
    db = DiskBloom(tmp_path, N=200_000, P=0.01, block_bits=1 << 12)
    B = db.meta.B  # number of blocks

    # Create keys targeting first ~min(16,B) blocks evenly
    blocks = min(16, B)
    per_block = 1500
    batches = [
        _force_keys_for_block(db, bid, per_block) for bid in range(blocks)
    ]
    flat = [k for batch in batches for k in batch]
    random.shuffle(flat)

    # 10 writer threads; concurrent readers polling might_contain as writers proceed
    writers = [flat[i::10] for i in range(10)]
    stop = threading.Event()
    seen_counter = {'count': 0}

    def writer(part):
        for k in part:
            db.add(k)

    def reader_probe():
        # best-effort: ensures reads are lock-free and files readable while writing
        while not stop.is_set():
            # sample some keys and call might_contain; may be false before written
            for k in random.sample(flat, 32):
                _ = db.might_contain(k)  # must not crash / corrupt
            seen_counter['count'] += 1

    threads = [threading.Thread(target=writer, args=(w,)) for w in writers]
    r = threading.Thread(target=reader_probe)
    r.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    r.join()

    # Post-condition: all keys were inserted => no false negatives now
    for k in random.sample(flat, 500):
        assert db.might_contain(k)

    assert seen_counter['count'] > 0  # reader loop actually ran


def _mp_same_block_worker(root_str: str, keys: list[str]) -> None:
    '''Top-level function so it can be pickled when using 'spawn'.'''
    import os
    from pathlib import Path

    # keep per-proc polars threads modest to avoid thrash
    os.environ.setdefault('POLARS_MAX_THREADS', '2')

    from native_db._bloom import (
        DiskBloom,
    )  # import inside to avoid pytest collection side effects

    local = DiskBloom(Path(root_str))  # reopen via metadata.json
    local.add_many(keys)  # single RMW per targeted block


def test_multiprocess_same_block_locking(tmp_path):
    '''
    Use add_many() so each process performs a single RMW for the hot block.
    Spawn context + join timeouts keep the suite from hanging.
    '''
    os.environ.setdefault('POLARS_MAX_THREADS', '2')

    db = DiskBloom(tmp_path, N=60_000, P=0.01, block_bits=1 << 12)
    bid = 1

    # 4 processes * 1000 keys, all mapped to the same block
    keys = _force_keys_for_block(db, bid, 4_000)
    shards = [keys[i::4] for i in range(4)]

    ctx = mp.get_context('spawn')
    procs = [
        ctx.Process(target=_mp_same_block_worker, args=(str(tmp_path), sh))
        for sh in shards
    ]
    for p in procs:
        p.start()

    # Bounded join; terminate on overrun
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(5)
            pytest.fail(f'child process {p.pid} did not exit in time')
        assert p.exitcode == 0

    # Post-condition: no false negatives for inserted keys
    import random

    for k in random.sample(keys, 300):
        assert db.might_contain(k)


def test_add_many_concurrent_with_add(tmp_path: Path):
    '''
    Interleave add_many (grouped per block) with single add() writers. Ensures
    both paths use per-block locking and produce a consistent result.
    '''
    db = DiskBloom(tmp_path, N=150_000, P=0.01, block_bits=1 << 12)

    # random keys for add_many; plus targeted same-block keys for add()
    rng = random.Random(42)
    bulk = [f'bulk-{i}-{rng.randrange(10**9)}' for i in range(8000)]
    bid = 2
    hot = _force_keys_for_block(db, bid, 4000)

    def t_bulk():
        db.add_many(bulk)  # single RMW per block

    def t_hot():
        for k in hot:
            db.add(k)  # RMW same block many times

    t1 = threading.Thread(target=t_bulk)
    t2 = threading.Thread(target=t_hot)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # spot-check membership
    for k in random.sample(bulk, 200):
        assert db.might_contain(k)
    for k in random.sample(hot, 200):
        assert db.might_contain(k)


def test_empirical_false_positive_rate(tmp_path: Path):
    '''
    Empirically estimate FP rate <= ~3x target P after inserting N distinct keys.
    We keep runtime modest but enough to be stable.
    '''
    N, P = 20_000, 0.01
    db = DiskBloom(tmp_path, N=N, P=P, block_bits=1 << 16)

    inserted = [f'in-{i}' for i in range(N)]
    db.add_many(inserted)

    # Generate non-members; avoid overlap
    probes = [f'out-{i}' for i in range(30_000)]

    # Count false positives
    fp = sum(1 for k in probes if db.might_contain(k))
    rate = fp / len(probes)

    # Very generous upper bound to prevent flakiness while still catching gross regressions
    assert rate <= max(0.03, 3 * P), (
        f'FP rate too high: {rate:.4f} (target {P})'
    )

    # Sanity: all inserted should be present (no false negatives)
    for k in random.sample(inserted, 300):
        assert db.might_contain(k)


def test_might_contain_many_basic(tmp_path: Path):
    # modest capacity to keep FP rate low while focusing on no-FN guarantee
    N, P = 10_000, 0.005
    db = DiskBloom(tmp_path, N=N, P=P, block_bits=1 << 15)

    # Insert a set of keys
    inserted = [f'ins-{i}' for i in range(3000)]
    db.add_many(inserted)

    # Build probes mixing: inserted (str), non-inserted (str), and bytes
    non = [f'non-{i}' for i in range(4000)]
    mixed = [
        inserted[0],
        non[0].encode(),
        inserted[1],
        non[1],
        inserted[2].encode(),
    ]
    # also check empty call
    assert db.might_contain_many([]) == []

    # Order must be preserved and all inserted must be True (no false negatives)
    out = db.might_contain_many(mixed)
    assert out[0] is True and out[2] is True and out[4] is True

    # Non-members may be false positives; just assert we get at least one False
    # so we know the function isn't trivially returning all True.
    assert any(not x for x in out), 'suspicious: all True for mixed set'

    # Larger batch: all inserted must be True
    res_inserted = db.might_contain_many(inserted)
    assert all(res_inserted), 'no false negatives allowed for inserted keys'

    # Larger batch: mostly non-members — FP rate should be small (generous bound)
    res_non = db.might_contain_many(non)
    fp_rate = sum(res_non) / len(res_non)
    assert fp_rate <= max(0.03, 3 * P)


def test_might_contain_many_reads_each_block_once(tmp_path: Path, monkeypatch):
    db = DiskBloom(tmp_path, N=50_000, P=0.01, block_bits=1 << 12)
    # Create keys targeting multiple known blocks
    blocks = min(8, db.meta.B)
    per_block = 50
    batches = [
        _force_keys_for_block(db, bid, per_block) for bid in range(blocks)
    ]
    keys = [k for batch in batches for k in batch]

    # Cold cache: ensure we haven't touched blocks yet
    # Count how many times the disk-read function runs
    calls = {'n': 0}
    original = db._read_bits_from_disk

    def counting_read(bid: int):
        calls['n'] += 1
        return original(bid)

    monkeypatch.setattr(db, '_read_bits_from_disk', counting_read)

    # Run vectorized probe (all blocks in one call)
    out = db.might_contain_many(keys)
    assert len(out) == len(keys)

    # Expect exactly one read per distinct block
    assert calls['n'] == blocks, (
        f'expected {blocks} block reads, saw {calls["n"]}'
    )


def test_might_contain_many_concurrent_with_writers(tmp_path: Path):
    db = DiskBloom(tmp_path, N=120_000, P=0.01, block_bits=1 << 12)

    # Prepare: some hot-block keys + random bulk for writers
    hot_bid = 0
    hot = _force_keys_for_block(db, hot_bid, 5000)
    rng = random.Random(123)
    bulk = [f'bulk-{i}-{rng.randrange(10**9)}' for i in range(8000)]

    # Reader probes (batched)
    mixed = hot[:1000] + [
        f'probe-{i}-{rng.randrange(10**9)}' for i in range(2000)
    ]

    stop = threading.Event()
    seen = {'loops': 0}

    def writer1():
        db.add_many(hot)

    def writer2():
        for k in bulk:
            db.add(k)

    def reader():
        while not stop.is_set():
            _ = db.might_contain_many(mixed)  # must not crash; lock-free reads
            seen['loops'] += 1

    t1 = threading.Thread(target=writer1)
    t2 = threading.Thread(target=writer2)
    r = threading.Thread(target=reader)
    r.start()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    stop.set()
    r.join()

    assert seen['loops'] > 0
    # Post condition: inserted keys should be present (no FNs)
    for k in random.sample(hot + bulk, 400):
        assert db.might_contain(k)
