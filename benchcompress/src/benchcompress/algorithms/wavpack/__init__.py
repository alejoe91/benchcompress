import numpy as np
import os


SOURCE_FILE = "wavpack/__init__.py"


def _load_long_description():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, "wavpack.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


LONG_DESCRIPTION = _load_long_description()


def wavpack_encode(x: np.ndarray, level: int) -> bytes:
    from wavpack_numcodecs import WavPack

    assert x.ndim == 1 or x.ndim == 2
    wv = WavPack(level=level)
    compressed = wv.encode(x)
    return compressed


def wavpack_decode(x: bytes, dtype: str, shape: tuple) -> np.ndarray:
    from wavpack_numcodecs import WavPack

    wv = WavPack()
    buf = wv.decode(x, dtype=dtype, shape=shape)
    y = np.frombuffer(buf, dtype=dtype)
    return y.reshape(shape)


algorithms = [
    {
        "name": "wavpack-1",
        "version": "1",
        "encode": lambda x: wavpack_encode(x, level=1),
        "decode": lambda x, dtype, shape: wavpack_decode(x, dtype, shape),
        "description": "WavPack compression at level 1, minimum compression level.",
        "tags": ["wavpack"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "wavpack-2",
        "version": "1",
        "encode": lambda x: wavpack_encode(x, level=2),
        "decode": lambda x, dtype, shape: wavpack_decode(x, dtype, shape),
        "description": "WavPack compression at level 2, medium compression level.",
        "tags": ["wavpack"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "wavpack-3",
        "version": "1",
        "encode": lambda x: wavpack_encode(x, level=3),
        "decode": lambda x, dtype, shape: wavpack_decode(x, dtype, shape),
        "description": "WavPack compression at level 3, high compression level.",
        "tags": ["wavpack"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "wavpack-4",
        "version": "1",
        "encode": lambda x: wavpack_encode(x, level=4),
        "decode": lambda x, dtype, shape: wavpack_decode(x, dtype, shape),
        "description": "WavPack compression at level 4, highest compression level.",
        "tags": ["wavpack"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
]
