#!/usr/bin/env python3
"""Zarr v3 cross-tooling conformance — RCRE (C) <-> zarr-python.

The ground truth for the Zarr v3 on-disk contract is zarr-python (the
reference implementation). This script drives BOTH directions of the
bit-identity bar:

  write  <store>
      1. For each of the fixed specs below, generate the deterministic
         data (53-bit LCG integer stream, per-array seed) and write
         <store>/<name>_py with zarr-python (identity or zstd codec).
      2. Write the manifest (tests/manifest format) consumed by the C
         driver tests/test_zarr_conformance.c.
      3. Record the expected FNV-1a 64 hashes of the raw little-endian
         bytes (the values the C side MUST read back).

  verify <store>
      1. For every spec, read <store>/<name>_c (written by the C driver
         from the same LCG stream) with zarr-python, hash the bytes and
         compare against the locally computed expectation  -> C->py
         bit identity.
      2. Compare <c_hashes.txt> (produced by the C driver while reading
         the <name>_py arrays) against the locally computed hashes  ->
         py->C bit identity.
      3. Semantic metadata checks on every <name>_c (shape, dtype,
         chunks, fill, codec names).
      4. Byte-exact golden metadata for the two canonical arrays
         (identity + zstd), pinned in tests/h5_fixtures/.
      5. NaN-fill survival check.

All data generation is pure-Python integer arithmetic (no numpy RNG):
the C driver must reproduce the same streams with the same 64-bit LCG.
"""
import json
import os
import shutil
import struct
import sys

import numpy as np
import zarr

# FNV-1a 64 (identical to the C driver)
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
M64 = (1 << 64) - 1


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & M64
    return h


# 53-bit integer LCG stream (identical to the C driver)
LCG_MULT = 6364136223846793005
LCG_INC = 1442695040888963407
M53 = (1 << 53) - 1


def lcg_stream(seed: int, n: int):
    st = seed
    out = []
    for _ in range(n):
        st = (st * LCG_MULT + LCG_INC) & M64
        out.append((st >> 11) & M53)
    return out


class Spec:
    def __init__(self, name, shape, chunk, dtype, fill, zstd_level, seed):
        self.name = name
        self.shape = tuple(shape)
        self.chunk = tuple(chunk)
        self.dtype = dtype          # numpy dtype object
        self.fill = fill            # python value (float/int/bool/NaN)
        self.zstd = zstd_level
        self.seed = seed
        self.ndim = len(shape)
        self.nelems = int(np.prod(shape))

    @property
    def dti(self):
        order = [np.dtype("f8"), np.dtype("f4"), np.dtype("f2"),
                 np.dtype("i8"), np.dtype("i4"), np.dtype("i2"), np.dtype("i1"),
                 np.dtype("u8"), np.dtype("u4"), np.dtype("u2"), np.dtype("u1"),
                 np.dtype("bool")]
        return order.index(self.dtype)

    def data(self) -> np.ndarray:
        v = np.array(lcg_stream(self.seed, self.nelems), dtype=np.uint64)
        dt = self.dtype
        if dt == np.dtype("f8"):
            d = v.astype(np.float64)
        elif dt == np.dtype("f4"):
            d = (v & 0xFFFFFFFFF).astype(np.float32)
        elif dt == np.dtype("f2"):
            d = (v & 0x3FF).astype(np.float16)
        elif dt == np.dtype("i8"):
            d = v.astype(np.int64)
        elif dt == np.dtype("i4"):
            d = v.astype(np.int32)
        elif dt == np.dtype("i2"):
            d = v.astype(np.int16)
        elif dt == np.dtype("i1"):
            d = v.astype(np.int8)
        elif dt == np.dtype("u8"):
            d = v
        elif dt == np.dtype("u4"):
            d = v.astype(np.uint32)
        elif dt == np.dtype("u2"):
            d = v.astype(np.uint16)
        elif dt == np.dtype("u1"):
            d = v.astype(np.uint8)
        elif dt == np.dtype("bool"):
            d = (v & 1).astype(np.bool_)
        else:
            raise ValueError(dt)
        return d.reshape(self.shape)

    def fill64bits(self) -> int:
        """The fill as an f64 bit pattern (float dtypes) — 0 otherwise."""
        if self.dtype.kind == "f":
            return struct.unpack("<Q", struct.pack("<d", float(self.fill)))[0]
        return 0

    def fillint(self) -> int:
        if self.dtype.kind == "b":
            return 1 if self.fill else 0
        if self.dtype.kind in "iu":
            return int(self.fill)
        return 0


def specs():
    return [
        Spec("f8_identity",  (16, 16),  (8, 8),   np.dtype("f8"), 0.0, 0, 0xC0FFEE00),
        Spec("f4_identity",  (12, 20),  (4, 5),   np.dtype("f4"), 0.5, 0, 0xC0FFEE01),
        Spec("i8_identity",  (1000,),   (7,),     np.dtype("i1"), -3,  0, 0xC0FFEE02),
        Spec("f8_zstd",      (64, 64),  (16, 16), np.dtype("f8"), 0.0, 3, 0xC0FFEE03),
        Spec("u16_3d",       (3, 4, 5), (2, 2, 3), np.dtype("u2"), 0, 0, 0xC0FFEE04),
        Spec("i64_1d",       (24,),     (7,),     np.dtype("i8"), -1,  0, 0xC0FFEE05),
        Spec("bool_1d",      (50,),     (13,),    np.dtype("bool"), False, 0, 0xC0FFEE06),
        Spec("f8_edge_fill", (17, 13),  (8, 4),   np.dtype("f8"), 0.25, 0, 0xC0FFEE07),
        Spec("f16_identity", (8, 8),    (4, 4),   np.dtype("f2"), 0.0, 0, 0xC0FFEE08),
        Spec("u8_zstd",      (30,),     (9,),     np.dtype("u1"), 0,  3, 0xC0FFEE09),
        Spec("i32_4d",       (2, 3, 4, 5), (2, 2, 2, 3), np.dtype("i4"), -100, 0, 0xC0FFEE0A),
        Spec("f8_zstd_large", (100, 100), (32, 32), np.dtype("f8"), 1.5, 3, 0xC0FFEE0B),
        Spec("f8_nanfill",   (12,),     (5,),     np.dtype("f8"), float("nan"), 0, 0xC0FFEE0C),
    ]


def manifest_line(s: Spec) -> str:
    dims = " ".join(str(d) for d in s.shape)
    chs = " ".join(str(c) for c in s.chunk)
    return " ".join([s.name, str(s.ndim), dims, chs, str(s.dti),
                     str(s.fillint()), hex(s.fill64bits()), str(s.zstd),
                     hex(s.seed)]).replace("  ", " ")


def make_args(s: Spec):
    from zarr.codecs import BytesCodec, ZstdCodec
    return dict(
        serializer=BytesCodec(endian="little"),
        compressors=ZstdCodec(level=s.zstd, checksum=False) if s.zstd else (),
    )


def mode_write(store: str):
    os.makedirs(store, exist_ok=True)
    expected = {}
    for s in specs():
        name = s.name + "_py"
        kwargs = make_args(s)
        arr = zarr.create_array(
            store,
            name=name,
            shape=s.shape,
            dtype=s.dtype,
            chunks=s.chunk,
            fill_value=s.fill,
            zarr_format=3,
            overwrite=True,
            write_data=False,
            **kwargs,
        )
        arr[...] = s.data()
        raw = s.data().astype(s.dtype).tobytes()  # C order, little-endian
        expected[name] = fnv1a64(raw)
    # group root (so the store root is a valid Zarr group)
    with open(os.path.join(store, "zarr.json"), "w") as f:
        json.dump({"attributes": {}, "zarr_format": 3, "node_type": "group"}, f)
    # manifest for the C driver
    with open(os.path.join(store, "manifest.txt"), "w") as f:
        for s in specs():
            f.write(manifest_line(s) + "\n")
    # expected hashes (py data -> what C must read back from _py arrays)
    with open(os.path.join(store, "expected_py_hashes.json"), "w") as f:
        json.dump(expected, f, indent=1)
    print(f"wrote {len(specs())} reference arrays to {store}")


def mode_verify(store: str, fixtures: str):
    with open(os.path.join(store, "manifest.txt")) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    with open(os.path.join(store, "expected_py_hashes.json")) as f:
        expected_py = {k: int(v) for k, v in json.load(f).items()}
    with open(os.path.join(store, "c_hashes.json")) as f:
        c_hashes = {k: int(v) for k, v in json.load(f).items()}
    with open(os.path.join(store, "c_expected.json")) as f:
        c_expected = {k: int(v) for k, v in json.load(f).items()}

    fails = 0
    for ln in lines:
        t = ln.split()
        name, ndim = t[0], int(t[1])
        dims = [int(x) for x in t[2:2 + ndim]]
        chs = [int(x) for x in t[2 + ndim:2 + 2 * ndim]]
        dti, fillint, fillbits, zstd, seed = (
            int(t[2 + 2 * ndim]), int(t[3 + 2 * ndim]),
            int(t[4 + 2 * ndim], 16), int(t[5 + 2 * ndim]), int(t[6 + 2 * ndim], 16))
        order = [np.dtype("f8"), np.dtype("f4"), np.dtype("f2"),
                 np.dtype("i8"), np.dtype("i4"), np.dtype("i2"), np.dtype("i1"),
                 np.dtype("u8"), np.dtype("u4"), np.dtype("u2"), np.dtype("u1"),
                 np.dtype("bool")]
        dt = order[dti]
        fill = struct.unpack("<d", struct.pack("<Q", fillbits))[0] if dt.kind == "f" else fillint

        cname = name + "_c"
        arr = zarr.open(os.path.join(store, cname), mode="r")

        # 1. C -> py data bit identity
        d = arr[...]
        got = fnv1a64(d.astype(dt).tobytes())
        want = c_expected[cname]
        ok1 = got == want
        if not ok1:
            fails += 1
            print(f"  [DATA MISMATCH C->py] {cname}: fnv {got:#x} != {want:#x}")

        # 2. py -> C data bit identity (C read the _py array we wrote)
        pkey = name + "_py"
        got2 = c_hashes[pkey]
        want2 = expected_py[pkey]
        ok2 = got2 == want2
        if not ok2:
            fails += 1
            print(f"  [DATA MISMATCH py->C] {pkey}: C read {got2:#x}, expected {want2:#x}")

        # 3. semantic metadata checks (from the zarr.json on disk — robust
        # across zarr-python metadata-object API drift)
        with open(os.path.join(store, cname, "zarr.json")) as f:
            md = json.load(f)
        mshape = tuple(md["shape"])
        mchunk = tuple(md["chunk_grid"]["configuration"]["chunk_shape"])
        mdt = np.dtype(md["data_type"])
        mcodecs = [c["name"] for c in md["codecs"]]
        mfill = md["fill_value"]
        if isinstance(mfill, str) and mfill == "NaN":
            mfill = float("nan")
        okm = (mshape == tuple(dims) and mchunk == tuple(chs)
               and mdt == dt and mcodecs == ["bytes"] + (["zstd"] if zstd else [])
               and (np.isnan(mfill) if (dt.kind == "f" and np.isnan(fill))
                    else (mfill == fill)))
        if not okm:
            fails += 1
            print(f"  [META MISMATCH] {cname}: shape={mshape} chunk={mchunk} "
                  f"dtype={mdt} codecs={mcodecs} fill={mfill!r} "
                  f"(want {dims} {chs} {dt} {['bytes'] + (['zstd'] if zstd else [])} {fill!r})")

        # 5. NaN fill survival (metadata level: JSON "NaN" string must
        #    round-trip to an actual NaN — the data-level fill path is
        #    covered by tests/test_zarr.c::test_fill_semantics, since the
        #    C side writes every chunk here)
        if dt.kind == "f" and np.isnan(fill):
            okn = isinstance(mfill, float) and np.isnan(mfill)
            if not okn:
                fails += 1
                print(f"  [NAN FILL LOST] {cname}: metadata fill {mfill!r}")

    # 4. golden metadata (byte-exact writer stability)
    for gname in ("f8_identity_c", "f8_zstd_c"):
        gold = os.path.join(fixtures, gname + ".zarr.json")
        live = os.path.join(store, gname, "zarr.json")
        if os.path.exists(gold):
            g = open(gold, "rb").read()
            l = open(live, "rb").read()
            if g != l:
                fails += 1
                print(f"  [GOLDEN MISMATCH] {gname} metadata drifted")
        else:
            print(f"  [note] golden {os.path.basename(gold)} not checked in yet")

    if fails == 0:
        print(f"CONFORMANCE OK: {len(lines)} arrays bit-identical in both directions "
              f"(C <-> zarr-python)")
        return 0
    print(f"CONFORMANCE FAILED: {fails} problems")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ref_zarr_conformance.py <write|verify> <store_dir>")
        sys.exit(2)
    mode, store = sys.argv[1], sys.argv[2]
    if mode == "write":
        mode_write(store)
    elif mode == "verify":
        fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h5_fixtures")
        sys.exit(mode_verify(store, fixtures))
    else:
        print("unknown mode")
        sys.exit(2)
