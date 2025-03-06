# Blosc2 Algorithm

Blosc2 is a modern, fast data compression library that builds upon the original Blosc library. It is designed for efficient compression of binary data, particularly optimized for in-memory compression of numerical arrays. Blosc2 uses block-oriented compression with support for multithreading and SIMD instructions.

## Features

- Fast compression and decompression speeds
- Block-oriented compression for better cache usage
- Support for various shuffling filters to improve compression ratios
- Built-in support for delta filtering
- Uses ZSTD compression codec

## Variants

### Standard Compression
Different compression levels trading off speed vs compression ratio:
- blosc2-1: Fastest compression with level 1
- blosc2-5: Balanced speed/compression with level 5
- blosc2-9: Maximum compression with level 9

### Advanced Variants

#### Delta Encoding (blosc2-9-delta)
Uses Blosc2's delta filter along with maximum compression. The delta filter stores differences between consecutive values, which is particularly effective for time series data where adjacent values are similar. This variant combines:
- Delta filtering for temporal correlation
- Bit shuffling for improved compression
- Level 9 compression for maximum compression ratio
