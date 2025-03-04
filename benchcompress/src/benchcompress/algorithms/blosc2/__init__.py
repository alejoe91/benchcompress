import numpy as np
import os
from ..ans.markov_reconstruct import markov_reconstruct as markov_reconstruct_cpp
from ..ans.markov_predict import markov_predict as markov_predict_cpp
from ..ans.get_run_lengths import get_run_lengths

SOURCE_FILE = "blosc2/__init__.py"


def _load_long_description():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, "blosc2.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


LONG_DESCRIPTION = _load_long_description()


def blosc2_encode(x: np.ndarray, clevel: int, filter: int = 2) -> bytes:
    import blosc2

    # Convert filter int to proper enum
    if filter == 2:
        blosc_filter = blosc2.Filter.BITSHUFFLE
    elif filter == 1:
        blosc_filter = blosc2.Filter.SHUFFLE
    else:
        blosc_filter = blosc2.Filter.NOFILTER

    # Get typesize from numpy array
    typesize = x.dtype.itemsize

    # Compress data
    compressed = blosc2.compress(
        x,  # numpy arrays support buffer interface
        typesize=typesize,
        clevel=clevel,
        filter=blosc_filter,
        codec=blosc2.Codec.ZSTD,
    )
    assert isinstance(compressed, bytes)  # Type assertion
    return compressed


def blosc2_decode(x: bytes, dtype: str, shape: tuple) -> np.ndarray:
    import blosc2

    decompressed = blosc2.decompress(x)
    assert isinstance(decompressed, (bytes, bytearray))  # Type assertion
    arr = np.frombuffer(decompressed, dtype=np.dtype(dtype))
    return arr.reshape(shape)


def blosc2_delta_encode(x: np.ndarray, clevel: int) -> bytes:
    import blosc2

    assert x.ndim == 1

    y = np.diff(x)
    y = np.insert(y, 0, x[0])
    buf = y.tobytes()

    # Convert default filter int to proper enum
    blosc_filter = blosc2.Filter.BITSHUFFLE

    # Get typesize from numpy array
    typesize = y.dtype.itemsize

    # Compress data
    compressed = blosc2.compress(
        buf,
        typesize=typesize,
        clevel=clevel,
        filter=blosc_filter,
        codec=blosc2.Codec.ZSTD,
    )
    assert isinstance(compressed, bytes)  # Type assertion
    return compressed


def blosc2_delta_decode(x: bytes, dtype: str, shape: tuple) -> np.ndarray:
    import blosc2

    assert len(shape) == 1

    decompressed = blosc2.decompress(x)
    assert isinstance(decompressed, (bytes, bytearray))  # Type assertion
    y = np.frombuffer(decompressed, dtype=np.dtype(dtype))
    return np.cumsum(y)


def blosc2_markov_encode(x: np.ndarray, clevel: int) -> bytes:
    import blosc2
    import struct

    assert x.ndim == 1
    coeffs, initial, resid = markov_predict_cpp(x, M=6, num_training_samples=10000)

    # Convert coeffs and initial to bytes
    coeffs_bytes = coeffs.tobytes()
    initial_bytes = initial.tobytes()

    # Create header with lengths
    header = struct.pack("QQ", len(coeffs_bytes), len(initial_bytes))

    # Convert default filter int to proper enum
    blosc_filter = blosc2.Filter.BITSHUFFLE

    # Get typesize from numpy array
    typesize = resid.dtype.itemsize

    # Compress data
    compressed = blosc2.compress(
        resid.tobytes(),
        typesize=typesize,
        clevel=clevel,
        filter=blosc_filter,
        codec=blosc2.Codec.ZSTD,
    )

    # Combine all parts
    return header + coeffs_bytes + initial_bytes + compressed


def blosc2_markov_decode(x: bytes, dtype: str, shape: tuple) -> np.ndarray:
    import blosc2
    import struct

    assert len(shape) == 1

    # Extract header
    header_size = struct.calcsize("QQ")
    coeffs_len, initial_len = struct.unpack("QQ", x[:header_size])

    # Extract coefficients and initial values
    pos = header_size
    coeffs = np.frombuffer(x[pos : pos + coeffs_len], dtype=np.float32)
    pos += coeffs_len
    initial = np.frombuffer(x[pos : pos + initial_len], dtype=dtype)
    pos += initial_len

    # Decompress residuals
    decompressed = blosc2.decompress(x[pos:])
    assert isinstance(decompressed, (bytes, bytearray))  # Type assertion
    resid = np.frombuffer(decompressed, dtype=dtype)

    # Reconstruct signal
    output = markov_reconstruct_cpp(coeffs, initial, resid)
    return output


def blosc2_markov_zrle_encode(x: np.ndarray, clevel: int) -> bytes:
    import blosc2
    import struct

    assert x.ndim == 1

    # Get run lengths for zero/non-zero sequences
    run_lengths = get_run_lengths(x)

    # Determine run length dtype code
    if run_lengths.dtype == np.uint8:
        run_length_dtype_code = 0
    elif run_lengths.dtype == np.uint16:
        run_length_dtype_code = 1
    elif run_lengths.dtype == np.uint32:
        run_length_dtype_code = 2
    else:
        raise ValueError(f"Unsupported run length dtype: {run_lengths.dtype}")

    # Extract non-zero data
    non_zero_arrays = []
    array_pos = 0
    i = 0
    while i < len(run_lengths):
        non_zero_len = int(run_lengths[i])
        if non_zero_len > 0:
            non_zero_arrays.append(x[array_pos : array_pos + non_zero_len])
            array_pos += non_zero_len
        i += 1
        if i < len(run_lengths):
            array_pos += int(run_lengths[i])  # Skip zeros
            i += 1

    non_zero_data = np.concatenate(non_zero_arrays)

    # Apply Markov prediction on non-zero data
    coeffs, initial, resid = markov_predict_cpp(
        non_zero_data, M=6, num_training_samples=10000
    )

    # Convert data to bytes
    coeffs_bytes = coeffs.tobytes()
    initial_bytes = initial.tobytes()
    run_lengths_bytes = run_lengths.tobytes()

    # Create header with lengths and dtype code
    header = struct.pack(
        "QQQQB",
        len(coeffs_bytes),
        len(initial_bytes),
        len(run_lengths_bytes),
        len(run_lengths),
        run_length_dtype_code,
    )

    # Convert default filter int to proper enum
    blosc_filter = blosc2.Filter.BITSHUFFLE

    # Get typesize from numpy array
    typesize = resid.dtype.itemsize

    # Compress residuals
    compressed = blosc2.compress(
        resid.tobytes(),
        typesize=typesize,
        clevel=clevel,
        filter=blosc_filter,
        codec=blosc2.Codec.ZSTD,
    )

    # Combine all parts
    return header + coeffs_bytes + initial_bytes + run_lengths_bytes + compressed


def blosc2_markov_zrle_decode(x: bytes, dtype: str, shape: tuple) -> np.ndarray:
    import blosc2
    import struct

    assert len(shape) == 1

    # Extract header
    header_size = struct.calcsize("QQQQB")
    coeffs_len, initial_len, run_lengths_len, num_run_lengths, run_length_dtype_code = (
        struct.unpack("QQQQB", x[:header_size])
    )

    # Extract components
    pos = header_size
    coeffs = np.frombuffer(x[pos : pos + coeffs_len], dtype=np.float32)
    pos += coeffs_len
    initial = np.frombuffer(x[pos : pos + initial_len], dtype=dtype)
    pos += initial_len

    # Get run lengths with proper dtype
    if run_length_dtype_code == 0:
        run_lengths = np.frombuffer(x[pos : pos + run_lengths_len], dtype=np.uint8)
    elif run_length_dtype_code == 1:
        run_lengths = np.frombuffer(x[pos : pos + run_lengths_len], dtype=np.uint16)
    elif run_length_dtype_code == 2:
        run_lengths = np.frombuffer(x[pos : pos + run_lengths_len], dtype=np.uint32)
    else:
        raise ValueError(f"Unsupported run length dtype code: {run_length_dtype_code}")
    pos += run_lengths_len

    assert len(run_lengths) == num_run_lengths

    # Decompress residuals
    decompressed = blosc2.decompress(x[pos:])
    assert isinstance(decompressed, (bytes, bytearray))  # Type assertion
    resid = np.frombuffer(decompressed, dtype=dtype)

    # Reconstruct non-zero data
    non_zero_data = markov_reconstruct_cpp(coeffs, initial, resid)

    # Reconstruct full array using run lengths
    non_zero_pos = 0
    i = 0
    segments = []
    while i < len(run_lengths):
        non_zero_len = int(run_lengths[i])
        if non_zero_len > 0:
            segment = non_zero_data[non_zero_pos : non_zero_pos + non_zero_len]
            segments.append(segment)
            non_zero_pos += non_zero_len
        i += 1
        if i < len(run_lengths):
            segments.append(np.zeros(int(run_lengths[i]), dtype=non_zero_data.dtype))
            i += 1

    return np.concatenate(segments)


algorithms = [
    {
        "name": "blosc2-1",
        "version": "1",
        "encode": lambda x: blosc2_encode(x, clevel=1),
        "decode": lambda x, dtype, shape: blosc2_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 1 (fastest compression).",
        "tags": ["blosc2"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-5",
        "version": "1",
        "encode": lambda x: blosc2_encode(x, clevel=5),
        "decode": lambda x, dtype, shape: blosc2_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 5 (balanced speed/compression).",
        "tags": ["blosc2"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-9",
        "version": "1",
        "encode": lambda x: blosc2_encode(x, clevel=9),
        "decode": lambda x, dtype, shape: blosc2_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 9 (maximum compression).",
        "tags": ["blosc2"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-9-delta",
        "version": "1",
        "encode": lambda x: blosc2_delta_encode(x, clevel=9),
        "decode": lambda x, dtype, shape: blosc2_delta_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 9 with delta encoding for improved compression of sequential data.",
        "tags": ["blosc2", "delta_encoding", "1d"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-1-markov",
        "version": "1",
        "encode": lambda x: blosc2_markov_encode(x, clevel=1),
        "decode": lambda x, dtype, shape: blosc2_markov_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 1 with Markov prediction for exploiting temporal correlations in the data.",
        "tags": ["blosc2", "markov_prediction", "1d"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-5-markov",
        "version": "1",
        "encode": lambda x: blosc2_markov_encode(x, clevel=1),
        "decode": lambda x, dtype, shape: blosc2_markov_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 5 with Markov prediction for exploiting temporal correlations in the data.",
        "tags": ["blosc2", "markov_prediction", "1d"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-9-markov",
        "version": "1",
        "encode": lambda x: blosc2_markov_encode(x, clevel=9),
        "decode": lambda x, dtype, shape: blosc2_markov_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 9 with Markov prediction for exploiting temporal correlations in the data.",
        "tags": ["blosc2", "markov_prediction", "1d"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "blosc2-9-markov-zrle",
        "version": "1",
        "encode": lambda x: blosc2_markov_zrle_encode(x, clevel=9),
        "decode": lambda x, dtype, shape: blosc2_markov_zrle_decode(x, dtype, shape),
        "description": "Blosc2 compression at level 9 with Markov prediction and zero run-length encoding for sparse data.",
        "tags": ["blosc2", "markov_prediction", "zero_rle", "1d"],
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
]
