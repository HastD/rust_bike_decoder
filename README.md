# BIKE decoder

This is a Rust implementation of the BGF decoding algorithm for QC-MDPC codes, which is the basis of the [BIKE key encapsulation mechanism](https://bikesuite.org/).

This implementation is intended as a research tool for investigating decoding failures and error floor phenomena in QC-MDPC codes; it is **not** designed to be cryptographically secure and does not follow all aspects of the BIKE specification. For an optimized implementation of the BIKE specification intended for cryptographic use, see [this constant-time software implementation](https://github.com/awslabs/bike-kem).

For a C implementation of BGF and several other decoding algorithms, see Valentin Vasseur's [qcmdpc_decoder](https://github.com/vvasseur/qcmdpc_decoder). An optimization in this code—the use of AVX2 instructions to speed up the inner loop of the bit-flipping algorithms by computing parity checks in parallel—is a direct Rust port of part of Vasseur's implementation.

## Compiling

To compile, you will need to [install Rust](https://www.rust-lang.org/tools/install) and run

```sh
cargo build --release
```

Compile-time parameters such as the block size and weight, the error vector weight, and the number of iterations in the BGF algorithm can be set in `src/parameters.rs`. The main executable will be generated at `target/release/bike_trials`.

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

A second executable used for analyzing absorbing sets will be generated at `target/release/absorbing`. This program accepts a list of decoding failures (in the JSON format generated by `bike_trials`) via `stdin` and filters the list to retain only those that correspond to absorbing sets on the Tanner graph.

This can be used efficiently in combination with a JSON-parsing utility like [jq](https://stedolan.github.io/jq/). For example, assuming the output of `bike_trials` is in `results.json`:

```sh
cat results.json | jq -c .decoding_failures | absorbing
```

(The call to `jq` extracts the field with key `"decoding_failures"` from the JSON input.)
