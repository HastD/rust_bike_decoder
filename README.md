# BIKE decoder

This is a Rust implementation of the BGF decoding algorithm for QC-MDPC codes, which is the basis of the [BIKE key encapsulation mechanism](https://bikesuite.org/).

This implementation is intended as a research tool for investigating decoding failures and error floor phenomena in QC-MDPC codes; it is **not** designed to be cryptographically secure and does not follow all aspects of the BIKE specification. For an optimized implementation of the BIKE specification intended for cryptographic use, see [this constant-time software implementation](https://github.com/awslabs/bike-kem).

For a C implementation of BGF and several other decoding algorithms, see Valentin Vasseur's [qcmdpc_decoder](https://github.com/vvasseur/qcmdpc_decoder). A key optimization in this code—the use of AVX2 instructions to speed up the inner loop of the bit-flipping algorithms by computing parity checks in parallel—is a direct Rust port of part of Vasseur's implementation.

## Compiling

To compile, you will need to [install Rust](https://www.rust-lang.org/tools/install) and run

```sh
cargo build --release
```

Compile-time parameters such as the block size and weight, the error vector weight, and the number of iterations in the BGF algorithm can be set in `src/parameters.rs`. The executable will be generated at `target/release/bike_decoder`.

By default, the program is built assuming your CPU supports the AVX2 instruction set. If this is not the case, remove the `rustflags` line in `.cargo/config.toml` before compiling; however, be aware that this program will run much more slowly without AVX2.

## Usage

```
Usage: bike_decoder [OPTIONS] --number <NUMBER>

Options:
  -N, --number <NUMBER>
          Number of trials (required)
  -w, --weak-keys <WEAK_KEYS>
          Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only) [default: 0]
  -T, --weak-key-threshold <WEAK_KEY_THRESHOLD>
          Weak key threshold [default: 3]
  -S, --atls <ATLS>
          Use error vectors from near-codeword set A_{t,l}(S) [possible values: C, N, 2N]
  -l, --atls-overlap <ATLS_OVERLAP>
          Overlap parameter l in A_{t,l}(S)
  -o, --output <OUTPUT>
          Output file [default stdout]
  -r, --recordmax <RECORDMAX>
          Max number of decoding failures recorded [default all]
  -s, --savefreq <SAVEFREQ>
          Save to disk frequency [default only at end]
      --threads <THREADS>
          Number of threads [default: 1]
  -v, --verbose
          Print decoding failures as they are found
  -h, --help
          Print help information
  -V, --version
          Print version information
```

The command-line program runs the following steps in a loop a number of times specified with the `-N` option:

1. Generate a random key of the specified length and weight.
2. Generate a random error vector of the specified weight.
3. Compute the syndrome of the error vector.
4. Use the BGF algorithm to attempt to decode the syndrome. Record any decoding failures.

The program outputs the resulting data in JSON format (either to a file specified with the `-o` option or to `stdout`). Additional options can be listed with the `--help` option, including filtering the keys to exclude certain classes of "weak key" or to generate *only* weak keys, limiting the number of decoding failures recorded, or running multiple threads at once. One particularly useful option for long-running trials is `-s`, which causes intermediate results to be written to disk, thus minimizing data loss if the program is interrupted.

Values for the `-N`, `-r`, and `-s` options can be given in scientific notation.

The `--atls` (or `-S`) and `--atls-overlap` (or `-l`) options, if either is provided, must both be given. They cause the error vectors to instead be generated from the sets of near-codewords `A_{t,l}(S)` described in Vasseur's thesis.

## Example 

To run 1 million trials and print the results in JSON format to the terminal:

```sh
bike_decoder -N 1000000
```

To run 100 million trials on non-weak keys only, with a weak key threshold of 4, saving the results to `results.json` every 1 million trials, using four threads and giving verbose output (printing decoding failures to `stdout` as they are found, and printing summary information at the beginning and end), run:

```sh
bike_decoder -N=1e8 -w=-1 -T=4 -o=results.json -s=1e6 --threads=4 -v
```
