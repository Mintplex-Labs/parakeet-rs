use std::path::PathBuf;
use std::{fmt, rc::Rc};

use crate::error::Result;
use ort::session::builder::SessionBuilder;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, MIGraphX) offer 5-10x speedup but require specific hardware.
// All GPU providers automatically fall back to CPU if they fail.
//
// Note: CoreML EP currently runs slower than CPU for Sortformer/Parakeet models because
// the ONNX graphs have dynamic input shapes, preventing CoreML from building optimised
// execution plans for ANE/GPU. CoreML claims nodes but runs them on CPU with overhead.
//
// WebGPU is experimental and may produce incorrect results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "migraphx")]
    MIGraphX,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
    #[cfg(feature = "nnapi")]
    NNAPI,
    #[cfg(feature = "vitisai")]
    VitisAI,
    #[cfg(feature = "qnn")]
    QNN,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub configure: Option<Rc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder>>>,
    /// Optional cache directory for compiled CoreML models. When set, avoids
    /// recompiling the ONNX-to-CoreML conversion on each session load (~5s).
    /// Only used when execution_provider is CoreML.
    pub coreml_cache_dir: Option<PathBuf>,
    /// VitisAI EP config file path (vai_ep_config.json).
    /// Only used when execution_provider is VitisAI.
    pub vitisai_config_file: Option<PathBuf>,
    /// VitisAI EP cache directory for compiled NPU models (provider-level cache).
    /// Only works with INT8 models when enable_cache_file_io_in_mem=0.
    pub vitisai_cache_dir: Option<PathBuf>,
    /// VitisAI EP cache key (subfolder name within cache_dir).
    pub vitisai_cache_key: Option<String>,
    /// Directory for ORT EP Context Cache files (_ctx.onnx).
    /// When set, from_pretrained will check for pre-compiled context models
    /// and generate them on first run. Works with all model types (FP32/INT8/BF16).
    pub ep_context_cache_dir: Option<PathBuf>,
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("execution_provider", &self.execution_provider)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field(
                "configure",
                &if self.configure.is_some() {
                    "<fn>"
                } else {
                    "None"
                },
            )
            .field("coreml_cache_dir", &self.coreml_cache_dir)
            .field("vitisai_config_file", &self.vitisai_config_file)
            .field("vitisai_cache_dir", &self.vitisai_cache_dir)
            .field("vitisai_cache_key", &self.vitisai_cache_key)
            .field("ep_context_cache_dir", &self.ep_context_cache_dir)
            .finish()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            configure: None,
            coreml_cache_dir: None,
            vitisai_config_file: None,
            vitisai_cache_dir: None,
            vitisai_cache_key: None,
            ep_context_cache_dir: None,
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_custom_configure(
        mut self,
        configure: impl Fn(SessionBuilder) -> ort::Result<SessionBuilder> + 'static,
    ) -> Self {
        self.configure = Some(Rc::new(configure));
        self
    }

    /// Set cache directory for compiled CoreML models.
    /// Avoids ~5s recompilation on each session load.
    pub fn with_coreml_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(path.into());
        self
    }

    /// Set VitisAI EP config file path (vai_ep_config.json).
    pub fn with_vitisai_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.vitisai_config_file = Some(path.into());
        self
    }

    /// Set VitisAI EP cache directory for compiled NPU models.
    pub fn with_vitisai_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.vitisai_cache_dir = Some(path.into());
        self
    }

    /// Set VitisAI EP cache key (subfolder name within cache_dir).
    pub fn with_vitisai_cache_key(mut self, key: impl Into<String>) -> Self {
        self.vitisai_cache_key = Some(key.into());
        self
    }

    /// Set directory for ORT EP Context Cache.
    /// On first load, compiled context models (_ctx.onnx) are dumped here.
    /// On subsequent loads, the context models are loaded directly, skipping compilation.
    pub fn with_ep_context_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.ep_context_cache_dir = Some(path.into());
        self
    }

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "webgpu",
            feature = "nnapi",
            feature = "vitisai",
            feature = "qnn"
        ))]
        use ort::ep::CPU as CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::ep::TensorRT::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::ep::coreml::{ComputeUnits, CoreML};
                let mut coreml = CoreML::default().with_compute_units(ComputeUnits::CPUAndGPU);

                if let Some(cache_dir) = &self.coreml_cache_dir {
                    coreml = coreml.with_model_cache_dir(cache_dir.to_string_lossy());
                }

                builder.with_execution_providers([
                    coreml.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX => builder.with_execution_providers([
                ort::ep::MIGraphX::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::ep::OpenVINO::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::ep::WebGPU::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI => builder.with_execution_providers([
                ort::ep::NNAPI::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "vitisai")]
            ExecutionProvider::VitisAI => {
                let mut vitis = ort::ep::Vitis::default();
                if let Some(config_file) = &self.vitisai_config_file {
                    vitis = vitis.with_config_file(config_file.to_string_lossy());
                }
                if let Some(cache_dir) = &self.vitisai_cache_dir {
                    vitis = vitis.with_cache_dir(cache_dir.to_string_lossy());
                }
                if let Some(cache_key) = &self.vitisai_cache_key {
                    vitis = vitis.with_cache_key(cache_key.as_str());
                }
                builder.with_execution_providers([
                    vitis.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "qnn")]
            ExecutionProvider::QNN => {
                use ort::ep::ArbitrarilyConfigurableExecutionProvider;
                let qnn = ort::ep::QNN::default()
                    .with_backend_path("QnnHtp.dll")
                    .with_arbitrary_config("htp_performance_mode", "burst")
                    .with_arbitrary_config("enable_htp_fp16_precision", "1")
                    .with_arbitrary_config("htp_graph_finalization_optimization_mode", "3");
                builder.with_execution_providers([
                    qnn.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }
        };

        if let Some(configure) = self.configure.as_ref() {
            builder = configure(builder)?;
        }

        Ok(builder)
    }
}
