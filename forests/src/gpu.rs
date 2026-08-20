//! GPU histogram accumulation with wgpu/WGSL (Issue #6).
//!
//! The expensive `record × feature` histogram build is dispatched as one
//! invocation per `(feature, record-slice)`. Each invocation owns a private
//! region of the partials buffer, so **no atomics are used** and every
//! partial is produced by a sequential, fixed-order loop. The CPU then folds
//! the partials in slice order in `f64`. The result is therefore
//! deterministic run-to-run; it differs from the CPU oracle only by `f32`
//! rounding inside a slice (documented tolerance: relative `1e-4` on gains),
//! and the final ranking always happens on the CPU via
//! [`crate::histogram::search_stumps`] with its deterministic tie-break.
//!
//! Buffer sizes are checked against the adapter limits before any allocation;
//! batches are split until they fit, so no record or feature is ever dropped.
//!
//! Without the `gpu` cargo feature this module compiles to a stub that
//! reports the GPU as unavailable; callers decide whether that is an error
//! (`--gpu on`) or a logged fallback (`--gpu auto`).
//!
//! **Measured economics (see `docs/benchmarks.md`):** on Apple-Silicon unified
//! memory the one-shot upload of the `u8` bin matrix dominates and the GPU path
//! is slower than the multi-threaded CPU path at production width, so the CLI
//! default is `--gpu off`. The kernel stays available for hosts where the
//! evidence differs.

use crate::config::GpuMode;
use crate::histogram::{ChunkSource, HistogramSet};

/// Which backend produced a histogram set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    /// CPU reference path.
    Cpu,
    /// GPU path; carries the adapter description.
    Gpu(String),
}

impl Backend {
    /// Journal label.
    pub fn label(&self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::Gpu(name) => format!("gpu:{name}"),
        }
    }
}

/// Records per GPU slice (each invocation loops over this many records).
pub const SLICE_RECORDS: usize = 8192;

/// Accumulate histograms honouring `mode`; returns the backend actually used.
/// `threads` bounds the CPU path's workers.
pub fn accumulate(
    mode: GpuMode,
    source: &dyn ChunkSource,
    bins_per_feature: &[usize],
    threads: usize,
) -> Result<(HistogramSet, Backend), String> {
    match mode {
        GpuMode::Off => Ok((
            HistogramSet::from_source_threads(source, bins_per_feature, threads)?,
            Backend::Cpu,
        )),
        GpuMode::On | GpuMode::Auto => match GpuAccumulator::new() {
            Ok(gpu) => {
                let name = gpu.adapter_name().to_string();
                Ok((
                    gpu.accumulate(source, bins_per_feature)?,
                    Backend::Gpu(name),
                ))
            }
            Err(e) if mode == GpuMode::Auto => {
                crate::log::warn(&format!(
                    "GPU unavailable ({e}); using CPU histogram search"
                ));
                Ok((
                    HistogramSet::from_source_threads(source, bins_per_feature, threads)?,
                    Backend::Cpu,
                ))
            }
            Err(e) => Err(format!("--gpu on but no GPU histogram backend: {e}")),
        },
    }
}

#[cfg(not(feature = "gpu"))]
mod imp {
    use super::*;

    /// GPU accumulator (stub: the crate was built without the `gpu` feature).
    pub struct GpuAccumulator;

    impl GpuAccumulator {
        /// Always fails: built without the `gpu` feature.
        pub fn new() -> Result<Self, String> {
            Err("neat_ai_forests was built without the `gpu` feature".into())
        }

        /// Unreachable in the stub.
        pub fn adapter_name(&self) -> &str {
            "none"
        }

        /// Unreachable in the stub.
        pub fn accumulate(
            &self,
            _source: &dyn ChunkSource,
            _bins: &[usize],
        ) -> Result<HistogramSet, String> {
            Err("neat_ai_forests was built without the `gpu` feature".into())
        }
    }
}

#[cfg(feature = "gpu")]
mod imp {
    use super::*;
    use crate::histogram::BinnedChunk;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    const SHADER: &str = r#"
struct Params {
    records: u32,
    features: u32,
    max_bins: u32,
    slice_records: u32,
    slices: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    let s = gid.y;
    if (f >= params.features || s >= params.slices) {
        return;
    }
    let start = s * params.slice_records;
    let end = min(start + params.slice_records, params.records);
    let base = (s * params.features + f) * params.max_bins;
    let section = params.slices * params.features * params.max_bins;
    for (var r = start; r < end; r = r + 1u) {
        let idx = r * params.features + f;
        let word = bins[idx >> 2u];
        var b = (word >> ((idx & 3u) * 8u)) & 0xffu;
        b = min(b, params.max_bins - 1u);
        let w = weight[r];
        let v = residual[r] * w;
        let o = base + b;
        out[o] = out[o] + w;
        out[section + o] = out[section + o] + v;
        out[2u * section + o] = out[2u * section + o] + v * residual[r];
    }
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        records: u32,
        features: u32,
        max_bins: u32,
        slice_records: u32,
        slices: u32,
        _pad: [u32; 3],
    }

    /// GPU accumulator bound to one adapter.
    pub struct GpuAccumulator {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        binding_limit: u64,
        name: String,
    }

    impl GpuAccumulator {
        /// Acquire a high-performance adapter or fail with a reason.
        pub fn new() -> Result<Self, String> {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))
                .map_err(|e| format!("no compatible GPU adapter: {e}"))?;
            let info = adapter.get_info();
            if matches!(
                info.backend,
                wgpu::Backend::Noop | wgpu::Backend::BrowserWebGpu
            ) {
                return Err(format!(
                    "adapter backend {:?} is not a native GPU",
                    info.backend
                ));
            }
            let limits = adapter.limits();
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("neat_ai_forests histogram device"),
                    required_limits: limits.clone(),
                    ..wgpu::DeviceDescriptor::default()
                }))
                .map_err(|e| format!("request_device failed: {e}"))?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("forests_histogram"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
            });
            let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("forests_histogram_layout"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("forests_histogram_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("forests_histogram_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            Ok(Self {
                device,
                queue,
                pipeline,
                layout,
                binding_limit: limits
                    .max_storage_buffer_binding_size
                    .min(limits.max_buffer_size),
                name: format!("{} ({:?})", info.name, info.backend),
            })
        }

        /// Adapter description.
        pub fn adapter_name(&self) -> &str {
            &self.name
        }

        /// Largest record batch whose buffers fit the adapter limits.
        pub fn batch_records(&self, features: usize, max_bins: usize) -> Result<usize, String> {
            let limit = self.binding_limit as usize;
            let by_bins = limit / features.max(1);
            // partials: 3 sections × slices × features × max_bins × 4 bytes
            let per_slice = 3 * features * max_bins * 4;
            if per_slice > limit {
                return Err(format!(
                    "features×bins partials ({per_slice} B) exceed the storage binding limit ({limit} B)"
                ));
            }
            let by_partials = (limit / per_slice) * SLICE_RECORDS;
            let batch = by_bins.min(by_partials);
            if batch == 0 {
                return Err("adapter limits too small for even one record".into());
            }
            Ok(batch.min(1 << 22))
        }

        /// Accumulate all chunks of `source`.
        pub fn accumulate(
            &self,
            source: &dyn ChunkSource,
            bins_per_feature: &[usize],
        ) -> Result<HistogramSet, String> {
            let features = source.features();
            let max_bins = bins_per_feature.iter().copied().max().unwrap_or(1);
            let mut set = HistogramSet::new(bins_per_feature);
            if features == 0 {
                return Ok(set);
            }
            let batch_records = self.batch_records(features, max_bins)?;
            let mut batch = BinnedChunk {
                features,
                ..Default::default()
            };
            let mut weights: Vec<f32> = Vec::new();
            let flush = |batch: &mut BinnedChunk,
                         weights: &mut Vec<f32>,
                         set: &mut HistogramSet|
             -> Result<(), String> {
                if batch.records == 0 {
                    return Ok(());
                }
                self.dispatch(batch, weights, max_bins, set)?;
                batch.records = 0;
                batch.bins.clear();
                batch.residual.clear();
                weights.clear();
                Ok(())
            };
            source.for_each_chunk(&mut |chunk| {
                let mut offset = 0;
                while offset < chunk.records {
                    let room = batch_records - batch.records;
                    let take = room.min(chunk.records - offset);
                    batch.bins.extend_from_slice(
                        &chunk.bins[offset * features..(offset + take) * features],
                    );
                    batch
                        .residual
                        .extend_from_slice(&chunk.residual[offset..offset + take]);
                    match &chunk.weight {
                        Some(w) => weights.extend_from_slice(&w[offset..offset + take]),
                        None => weights.extend(std::iter::repeat_n(1.0f32, take)),
                    }
                    batch.records += take;
                    offset += take;
                    if batch.records == batch_records {
                        flush(&mut batch, &mut weights, &mut set)?;
                    }
                }
                Ok(())
            })?;
            flush(&mut batch, &mut weights, &mut set)?;
            Ok(set)
        }

        fn dispatch(
            &self,
            batch: &BinnedChunk,
            weights: &[f32],
            max_bins: usize,
            set: &mut HistogramSet,
        ) -> Result<(), String> {
            let features = batch.features;
            let records = batch.records;
            let slices = records.div_ceil(SLICE_RECORDS);
            let partial_len = 3 * slices * features * max_bins;
            let partial_bytes = (partial_len * 4) as u64;
            let mut packed = batch.bins.clone();
            packed.resize(packed.len().div_ceil(4) * 4, 0);
            for (name, bytes) in [("bins", packed.len() as u64), ("partials", partial_bytes)] {
                if bytes > self.binding_limit {
                    return Err(format!(
                        "{name} buffer ({bytes} B) exceeds the storage binding limit ({} B)",
                        self.binding_limit
                    ));
                }
            }
            let params = Params {
                records: records as u32,
                features: features as u32,
                max_bins: max_bins as u32,
                slice_records: SLICE_RECORDS as u32,
                slices: slices as u32,
                _pad: [0; 3],
            };
            let mk = |label: &str, contents: &[u8], usage| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents,
                        usage,
                    })
            };
            let params_buf = mk(
                "params",
                bytemuck::bytes_of(&params),
                wgpu::BufferUsages::UNIFORM,
            );
            let bins_buf = mk("bins", &packed, wgpu::BufferUsages::STORAGE);
            let residual_buf = mk(
                "residual",
                bytemuck::cast_slice(&batch.residual),
                wgpu::BufferUsages::STORAGE,
            );
            let weight_buf = mk(
                "weight",
                bytemuck::cast_slice(weights),
                wgpu::BufferUsages::STORAGE,
            );
            let partials_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partials"),
                size: partial_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: partial_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forests_histogram_bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bins_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: residual_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: partials_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forests_histogram"),
                });
            encoder.clear_buffer(&partials_buf, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("histogram"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((features as u32).div_ceil(64), slices as u32, 1);
            }
            encoder.copy_buffer_to_buffer(&partials_buf, 0, &readback, 0, partial_bytes);
            self.queue.submit(Some(encoder.finish()));
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("device poll failed: {e:?}"))?;
            rx.recv()
                .map_err(|_| "map_async sender dropped".to_string())?
                .map_err(|e| format!("map_async failed: {e:?}"))?;
            let mapped = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&mapped);
            let section = slices * features * max_bins;
            // Fold partials in slice order (deterministic), in f64.
            for s in 0..slices {
                for f in 0..features {
                    let h = &mut set.features[f];
                    let bins = h.count.len();
                    let base = (s * features + f) * max_bins;
                    for b in 0..bins.min(max_bins) {
                        h.count[b] += f64::from(floats[base + b]);
                        h.sum[b] += f64::from(floats[section + base + b]);
                        h.sum_sq[b] += f64::from(floats[2 * section + base + b]);
                    }
                }
            }
            drop(mapped);
            readback.unmap();
            // Totals from the CPU-side residual/weight vectors (exact, f64).
            for (r, w) in batch.residual.iter().zip(weights) {
                let (r, w) = (f64::from(*r), f64::from(*w));
                set.total_count += w;
                set.total_sum += w * r;
                set.total_sum_sq += w * r * r;
            }
            set.records += records as u64;
            Ok(())
        }
    }
}

pub use imp::GpuAccumulator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_always_uses_cpu() {
        let src = crate::histogram::MemorySource {
            chunks: vec![],
            label: "t".into(),
        };
        let (set, backend) = accumulate(GpuMode::Off, &src, &[4], 2).unwrap();
        assert_eq!(backend, Backend::Cpu);
        assert_eq!(set.records, 0);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn on_mode_fails_clearly_without_feature() {
        let src = crate::histogram::MemorySource {
            chunks: vec![],
            label: "t".into(),
        };
        let err = accumulate(GpuMode::On, &src, &[4], 2).unwrap_err();
        assert!(err.contains("--gpu on") && err.contains("gpu"), "{err}");
        let (_, backend) = accumulate(GpuMode::Auto, &src, &[4], 2).unwrap();
        assert_eq!(backend, Backend::Cpu);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_matches_cpu_oracle_within_tolerance() {
        use crate::histogram::{
            BinnedChunk, HistogramSet, MemorySource, SearchControls, search_stumps,
        };
        let Ok(gpu) = GpuAccumulator::new() else {
            eprintln!("skipping GPU parity test: no adapter");
            return;
        };
        // 20_000 records × 37 features, ragged bin counts, weights on some chunks,
        // several chunks so batching/slicing paths are exercised.
        let features = 37;
        let bins_per_feature: Vec<usize> = (0..features).map(|f| 3 + (f * 7) % 30).collect();
        let mut seed = 99u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut chunks = Vec::new();
        for c in 0..7 {
            let records = 2000 + c * 300;
            let mut bins = Vec::with_capacity(records * features);
            let mut residual = Vec::with_capacity(records);
            let mut weight = Vec::with_capacity(records);
            for _ in 0..records {
                for &nb in bins_per_feature.iter().take(features) {
                    bins.push((next() as usize % nb) as u8);
                }
                // Planted signal on feature 3 so the top stump is unambiguous;
                // everything else is noise with near-tied gains.
                let planted = if bins[bins.len() - features + 3] > 10 {
                    0.5
                } else {
                    0.0
                };
                residual.push((next() as f32 / u32::MAX as f32) * 2.0 - 1.0 + planted);
                weight.push(1.0 + (next() % 3) as f32 * 0.5);
            }
            chunks.push(BinnedChunk {
                records,
                features,
                bins,
                residual,
                weight: if c % 2 == 0 { Some(weight) } else { None },
                first_index: 0,
            });
        }
        let src = MemorySource {
            chunks,
            label: "t".into(),
        };
        let cpu = HistogramSet::from_source(&src, &bins_per_feature).unwrap();
        let g1 = gpu.accumulate(&src, &bins_per_feature).unwrap();
        let g2 = gpu.accumulate(&src, &bins_per_feature).unwrap();
        assert_eq!(g1, g2, "GPU accumulation must be deterministic run-to-run");
        assert_eq!(g1.records, cpu.records);
        for (f, (a, b)) in cpu.features.iter().zip(&g1.features).enumerate() {
            for bin in 0..a.count.len() {
                assert!(
                    (a.count[bin] - b.count[bin]).abs() < 1e-3,
                    "count f{f} b{bin}: {} vs {}",
                    a.count[bin],
                    b.count[bin]
                );
                let tol = 1e-4 * a.sum[bin].abs().max(1.0);
                assert!(
                    (a.sum[bin] - b.sum[bin]).abs() < tol,
                    "sum f{f} b{bin}: {} vs {}",
                    a.sum[bin],
                    b.sum[bin]
                );
                let tol = 1e-4 * a.sum_sq[bin].abs().max(1.0);
                assert!(
                    (a.sum_sq[bin] - b.sum_sq[bin]).abs() < tol,
                    "sumsq f{f} b{bin}"
                );
            }
        }
        let controls = SearchControls {
            min_leaf_records: 10.0,
            top_k: 20,
            ..Default::default()
        };
        let thresholds = |_: usize, b: usize| Some(b as f32);
        let top_cpu = search_stumps(&cpu, &thresholds, &controls, "cpu");
        let top_gpu = search_stumps(&g1, &thresholds, &controls, "gpu");
        assert_eq!(top_cpu.len(), top_gpu.len());
        // The planted split must win on both backends; lower ranks may swap
        // among near-tied noise stumps (f32 rounding), but their gains agree.
        assert_eq!((top_cpu[0].feature, top_cpu[0].bin), (3, 10));
        assert_eq!((top_gpu[0].feature, top_gpu[0].bin), (3, 10));
        for (a, b) in top_cpu.iter().zip(&top_gpu) {
            assert!(
                (a.gain - b.gain).abs() <= 1e-4 * a.gain.abs().max(1.0),
                "{} vs {}",
                a.gain,
                b.gain
            );
        }
    }
}
