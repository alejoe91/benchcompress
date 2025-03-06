from .bzip2 import algorithms as bzip2_algorithms
from .zlib import algorithms as zlib_algorithms
from .zstd import algorithms as zstd_algorithms
from .ans import algorithms as ans_algorithms
from .lzma import algorithms as lzma_algorithms
from .brotli import algorithms as brotli_algorithms
from .lz4 import algorithms as lz4_algorithms
from .blosc2 import algorithms as blosc2_algorithms
from .wavpack impot algorithms as wavpack_algorithms

algorithms = (
    bzip2_algorithms
    + zlib_algorithms
    + zstd_algorithms
    + ans_algorithms
    + lzma_algorithms
    + brotli_algorithms
    + lz4_algorithms
    + blosc2_algorithms
    + wavpack_algorithms
)
