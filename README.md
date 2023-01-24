# BIKE decoder

This is a Rust implementation of the BGF decoding algorithm for QC-MDPC codes, which is the basis of the [BIKE key encapsulation mechanism](https://bikesuite.org/).

This implementation is intended as a research tool for investigating decoding failures and error floor phenomena in QC-MDPC codes; it is **not** designed to be cryptographically secure and does not follow all aspects of the BIKE specification. For an optimized implementation of the BIKE specification intended for cryptographic use, see [this constant-time software implementation](https://github.com/awslabs/bike-kem).

For a C implementation of BGF and several other decoding algorithms, see Valentin Vasseur's [qcmdpc_decoder](https://github.com/vvasseur/qcmdpc_decoder). An optimization in this code—the use of AVX2 instructions to speed up the inner loop of the bit-flipping algorithms by computing parity checks in parallel—is a direct Rust port of part of Vasseur's implementation.

## Compiling

To compile, you will need to [install Rust](https://www.rust-lang.org/tools/install) and run

```sh
cargo build --release
```

Compile-time parameters such as the block size and weight, the error vector weight, and the number of iterations in the BGF algorithm are defined in `src/parameters.rs`. Different values can be set at compile-time using the environment variables `BIKE_BLOCK_LENGTH`, `BIKE_BLOCK_WEIGHT`, `BIKE_ERROR_WEIGHT`, and `BIKE_NB_ITER`. The main executable will be generated at `target/release/bike_trials`.

## Usage

```
Usage: bike_trials [OPTIONS] --number <NUMBER>

Options:
  -N, --number <NUMBER>
          Number of trials (required)
  -w, --weak-keys <WEAK_KEYS>
          Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only) [default: 0]
  -T, --weak-key-threshold <WEAK_KEY_THRESHOLD>
          Weak key threshold [default: 3]
      --fixed-key <FIXED_KEY>
          Always use the specified key (in JSON format)
  -S, --ncw <NCW>
          Use error vectors from near-codeword set A_{t,l}(S) [possible values: C, N, 2N]
  -l, --ncw-overlap <NCW_OVERLAP>
          Overlap parameter l in A_{t,l}(S)
  -o, --output <OUTPUT>
          Output file [default: stdout]
      --overwrite
          If output file already exists, overwrite without creating backup
      --parallel
          Run in parallel with automatically chosen number of threads
  -r, --recordmax <RECORDMAX>
          Max number of decoding failures recorded [default: 10000]
  -s, --savefreq <SAVEFREQ>
          Save to disk frequency [default: only at end]
      --seed <SEED>
          Specify PRNG seed as 256-bit hex string [default: random]
      --seed-index <SEED_INDEX>
          Initialize PRNG to match specified thread index (single-threaded only)
      --threads <THREADS>
          Set number of threads (ignores --parallel)
  -v, --verbose...
          Print statistics and/or decoding failures [repeat for more verbose, max 3]
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

The program outputs the resulting data in JSON format, either to a file specified with the `-o` option or to `stdout`. If the specified output file already exists and is nonempty, it will be backed up by appending a random UUID to the filename unless the `--overwrite` flag is provided. If the `-o` option is not provided, the output to `stdout` will consist only of the JSON data (possibly multiple times if `--savefreq` is specified) and thus can be used with shell redirection operators (e.g. piping to another program that expects JSON input).

Additional options can be listed with the `--help` option, including filtering the keys to exclude certain classes of "weak key" or to generate *only* weak keys, limiting the number of decoding failures recorded, or running multiple threads at once. A useful option for long-running trials is `--savefreq`, which causes intermediate results to be written to disk, thus minimizing data loss if the program is interrupted.

Values for the `-N`, `--recordmax`, and `--savefreq` options can be given in scientific notation.

The `--ncw` (or `-S`) option causes the error vectors to instead be generated from the sets of near-codewords `A_{t,l}(S)` described in Vasseur's thesis. The overlap `l` with the specified set `S` can be fixed with the `--ncw-overlap` (or `-l`) parameter; if omitted, the overlap parameter will be chosen at random with each iteration.

## Examples

To run 1 million trials (with random keys and random error vectors) and print the results in JSON format to standard output:

```sh
bike_trials -N=1e6
```

To run 100 million trials in parallel on non-weak keys only, with a weak key threshold of 4, saving the results to `results.json` every 1 million trials, and printing summary information at the beginning and end:

```sh
bike_trials -N=1e8 -w=-1 -T=4 -o=results.json -s=1e6 --parallel -v
```

To run 25 million trials with a given fixed key (replace `...` with the support of the key blocks as a comma-separated list), using error vectors in the near-codeword set `A_{t,7}(N)`, recording a maximum of 100 decoding failures (additional decoding failures will be counted but the vectors will not be recorded), printing full verbose output, and saving the results to `results.json` at the end:

```sh
bike_trials -N=2.5e7 -o=results.json --fixed-key='{"h0": [...], "h1": [...]}' --recordmax=100 -S=N -l=7 -v -v -v
```

## Absorbing sets utility

A second executable used for analyzing absorbing sets will be generated at `target/release/absorbing`. This program has three subcommands: `filter`, `enumerate`, and `sample`. The `--parallel` flag may be specified (before the subcommand) to run the computations in parallel using multiple threads.

The `filter` command accepts a list of decoding failures (in the JSON format generated by `bike_trials`) via `stdin` and filters the list to retain only those that correspond to absorbing sets on the Tanner graph, attaching some metadata about the absorbing set. This can be used efficiently in combination with a JSON-parsing utility like [jq](https://stedolan.github.io/jq/). For example, assuming the output of `bike_trials` is in `results.json`:

```sh
cat results.json | jq -c .decoding_failures | absorbing filter
```

> Warning: The input to `absorbing filter` must have been generated using the same values of the constants `BLOCK_WEIGHT`, `BLOCK_LENGTH`, and `ERROR_WEIGHT` in `bike_decoder/src/parameters.rs`. If the values are different, parsing errors or incorrect output may occur.

The `enumerate` command exhaustively searches for absorbing sets of a given weight for a given key. The block weight and length of the key are specified at compile-time via the constants `BLOCK_WEIGHT` and `BLOCK_LENGTH` in `absorbing/src/main.rs`. The weight of the absorbing sets is a required argument `--weight`; the key is either randomly generated or can be specified (in JSON format) with the `--key` argument. Results are sent in JSON format to `stdout`. The `--verbose` flag prints summary information to `stderr`. Note that this may take a very long time if the parameters are not small.

The `sample` command is similiar to `enumerate`, but randomly samples a number of times determined by the `-N` option rather than exhaustively enumerating.

## Near-codeword classifier

Another executable, generated at `target/release/ncw-classify`, classifies vectors into near-codeword sets. It has two subcommands: `process` and `sample`. The `process` command accepts a list of decoding failures via `stdin` and classifies each of the error vector supports. The `sample` command uses randomly generated vectors of a specified weight in place of the input. The `--parallel` flag may be specified (before the subcommand) to run the computations in parallel using multiple threads.
