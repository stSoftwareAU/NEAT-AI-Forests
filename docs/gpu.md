# GPU histogram accumulation (Issue #6)

Enabled with `cargo build --features gpu`; selected at run time with `--gpu
auto|on|off`. Without the feature, or without a native adapter, `auto` logs
`GPU unavailable (…); using CPU histogram search` and the journal's
`searchBackend` says `cpu` — the program never pretends the GPU ran. `--gpu on`
fails instead of falling back.

## Kernel

One WGSL compute shader (`forests/src/gpu.rs`), workgroup size 64, dispatched
as `(ceil(features / 64), slices, 1)`. Invocation `(f, s)` loops sequentially
over the records of slice `s` (8192 records) and accumulates `count`, `Σ w·r`
and `Σ w·r²` for feature `f` into its **own** region of the partials buffer:

```text
partials[section][slice][feature][bin]   section ∈ {count, sum, sumsq}
```

No two invocations write the same address, so **no atomics** are used and each
partial is produced by a fixed-order loop. The CPU folds partials in slice
order in `f64`. The result is deterministic run to run (tested by accumulating
twice and asserting equality).

Inputs: the `u8` bin matrix packed four per `u32`, the `f32` residual vector and
an `f32` weight vector (all ones when the search set is unweighted).

## Agreement with the CPU oracle

Within a slice the sums are `f32`, so statistics differ from the `f64` CPU
reference by rounding only. The documented tolerance is relative `1e-4` on
per-bin sums and on stump gains; the parity test
(`gpu::tests::gpu_matches_cpu_oracle_within_tolerance`) checks every bin of a
ragged 37-feature fixture, that the planted top stump wins on both backends,
and that rank-wise gains agree. Near-tied noise stumps may swap positions
between backends; final ranking always happens on the CPU with the
deterministic tie-break, which is what the issue asked for.

## Limits

Before any allocation the accumulator computes the largest record batch whose
bin buffer and partials buffer both fit
`min(max_storage_buffer_binding_size, max_buffer_size)`; chunks are split and
dispatched repeatedly, so no record or feature is dropped. A configuration
where one slice's partials alone exceed the limit is an explicit error.

## Backends

`wgpu` with default features: Metal on macOS, Vulkan/DX12/GL elsewhere.
`Noop`/`BrowserWebGpu` adapters are treated as "no GPU". CI runners have no
GPU; the parity test prints a skip notice there.

## Economics — measured negative result

See [benchmarks.md](benchmarks.md). On an Apple M4 the GPU path takes ≈ 2.1 s
for 200 k × 2461 (upload dominated) against 0.35 s for the feature-split
8-thread CPU path — **0.17×**. `--gpu off` is therefore the default. Reusing
an uploaded matrix across tree levels, or a discrete GPU with the matrix
resident, are the conditions under which it would be worth re-measuring.
