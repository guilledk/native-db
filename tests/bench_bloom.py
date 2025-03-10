import os
import random
import shutil
import tempfile

from pathlib import Path

import pytest
import xxhash

from native_db._bloom import DiskBloom


os.environ.setdefault('POLARS_MAX_THREADS', '4')


@pytest.mark.benchmark(
    max_time=5,
    min_rounds=1,
)
def test_write_throughput(benchmark):
    tmp = Path(tempfile.mkdtemp())
    try:
        db = DiskBloom(tmp, N=200_000, P=0.01, block_bits=1 << 16)
        rng = random.Random(42)
        inserts = [f'ins-{i}-{rng.randrange(10**12)}' for i in range(10_000)]

        def run():
            db.add_many(inserts)

        benchmark(run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.benchmark(
    max_time=5,
    min_rounds=1,
)
def test_read_throughput(benchmark):
    tmp = Path(tempfile.mkdtemp())
    try:
        db = DiskBloom(tmp, N=200_000, P=0.01, block_bits=1 << 16)
        # prefill
        rng = random.Random(42)
        inserts = [f'ins-{i}-{rng.randrange(10**12)}' for i in range(10_000)]
        db.add_many(inserts)

        probes = [f'probe-{i}-{rng.randrange(10**12)}' for i in range(10_000)]

        def run():
            return sum(1 for k in probes if db.might_contain(k))

        benchmark(run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.benchmark(max_time=5, min_rounds=1)
def test_read_throughput_many(benchmark):
    tmp = Path(tempfile.mkdtemp())
    try:
        db = DiskBloom(tmp, N=200_000, P=0.01, block_bits=1 << 16)
        # prefill
        rng = random.Random(42)
        inserts = [f'ins-{i}-{rng.randrange(10**12)}' for i in range(10_000)]
        db.add_many(inserts)

        probes = [f'probe-{i}-{rng.randrange(10**12)}' for i in range(10_000)]

        def run():
            out = db.might_contain_many(probes)
            # return a cheap aggregate to prevent dead-code elimination
            return sum(out)

        benchmark(run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.benchmark(max_time=5, min_rounds=1)
def test_read_throughput_many_multiblock(benchmark):
    tmp = Path(tempfile.mkdtemp())
    try:
        db = DiskBloom(tmp, N=300_000, P=0.01, block_bits=1 << 12)
        # craft probes spanning several blocks evenly
        blocks = min(16, db.meta.B)
        per_block = 1000
        batches = []
        for bid in range(blocks):
            # re-use the same helper logic as tests to target block ids
            # (inline minimal variant to avoid importing test helpers)
            s1, _ = db.meta.seeds
            i = 0
            found = 0
            while found < per_block:
                k = f'probe-b{bid}-{i}'
                h1 = xxhash.xxh64_intdigest(k.encode(), seed=s1)
                if (h1 % db.meta.B) == bid:
                    batches.append(k)
                    found += 1
                i += 1

        probes = batches

        def run():
            out = db.might_contain_many(probes)
            return sum(out)

        benchmark(run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
