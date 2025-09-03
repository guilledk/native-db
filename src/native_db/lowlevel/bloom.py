import mmap
import math
import struct

from pathlib import Path

import xxhash


def _hash64(data: bytes, *, seed: int) -> int:
    '''
    Fast bytes -> int hashing function.

    '''
    return xxhash.xxh64_intdigest(data, seed=seed)


def bloom_params(n: int, p: float) -> tuple[int, int]:
    '''
    Given expected distinct items n and false-positive rate p, return (m bits, k hashes).

    '''
    if n <= 0 or not (0 < p < 1):
        raise ValueError('n must be > 0 and 0 < p < 1')
    m = math.ceil(-(n * math.log(p)) / (math.log(2) ** 2))
    k = max(1, round((m / n) * math.log(2)))
    return m, k


MAGIC = 0xB10F_B10F
HDR_FMT = "<I I Q Q I I"  # magic, version, m_bits, k, seed1, seed2
HDR_SIZE = struct.calcsize(HDR_FMT)
VERSION = 1


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


class DiskBloom:
    def __init__(self, path: str | Path, *, N: int, P: float,
                 seeds: tuple[int, int] = (0x12345678, 0x9ABCDEF0), create: bool = True):
        self.path = Path(path)
        m_raw, k = bloom_params(N, P)          # same math you use today
        m_bits = _next_pow2(m_raw)             # power of two for & mask
        self.k = k
        self.m_bits = m_bits
        self.mask = m_bits - 1
        self.seed1, self.seed2 = seeds

        file_size = HDR_SIZE + (m_bits // 8)

        # create or open & validate
        if create or not self.path.exists():
            with open(self.path, "wb") as f:
                f.truncate(file_size)
                hdr = struct.pack(HDR_FMT, MAGIC, VERSION, m_bits, k, self.seed1, self.seed2)
                f.seek(0); f.write(hdr)
        else:
            with open(self.path, "rb") as f:
                hdr = f.read(HDR_SIZE)
                magic, ver, m_bits_f, k_f, s1, s2 = struct.unpack(HDR_FMT, hdr)
                if magic != MAGIC or ver != VERSION:
                    raise RuntimeError("invalid bloom header")
                self.m_bits = m_bits_f
                self.k = k_f
                self.mask = self.m_bits - 1
                self.seed1, self.seed2 = s1, s2
                file_size = HDR_SIZE + (self.m_bits // 8)

        # mmap the bit array region
        self.fh = open(self.path, "r+b")
        if self.fh.tell() != file_size:
            self.fh.truncate(file_size)
        self.mm = mmap.mmap(self.fh.fileno(), length=0)  # whole file
        self.bits_off = HDR_SIZE  # start of bit array

    def _h1_h2(self, key: bytes | str) -> tuple[int, int]:
        b = key if isinstance(key, bytes) else key.encode("utf-8")
        return _hash64(b, seed=self.seed1), _hash64(b, seed=self.seed2)

    def add(self, key: bytes | str) -> None:
        h1, h2 = self._h1_h2(key)
        base = self.bits_off
        for i in range(self.k):
            bit = (h1 + i * h2) & self.mask
            byte = bit >> 3
            off  = bit & 7
            idx = base + byte
            cur = self.mm[idx]
            self.mm[idx] = cur | (1 << off)

    def add_many(self, keys) -> None:
        base = self.bits_off
        k = self.k; mask = self.mask
        for key in keys:
            b = key if isinstance(key, bytes) else key.encode("utf-8")
            h1 = _hash64(b, seed=self.seed1)
            h2 = _hash64(b, seed=self.seed2)
            for i in range(k):
                bit = (h1 + i * h2) & mask
                byte = bit >> 3
                off  = bit & 7
                idx = base + byte
                cur = self.mm[idx]
                self.mm[idx] = cur | (1 << off)

    def might_contain(self, key: bytes | str) -> bool:
        h1, h2 = self._h1_h2(key)
        base = self.bits_off
        for i in range(self.k):
            bit = (h1 + i * h2) & self.mask
            byte = bit >> 3
            off  = bit & 7
            if (self.mm[base + byte] & (1 << off)) == 0:
                return False
        return True

    def might_contain_many(self, keys):
        return [self.might_contain(k) for k in keys]

    def flush(self) -> None:
        self.mm.flush()

    def close(self) -> None:
        try:
            self.mm.flush()
        finally:
            self.mm.close()
            self.fh.close()
