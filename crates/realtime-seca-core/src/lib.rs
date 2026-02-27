pub mod config;
pub mod engine;
pub mod error;
pub mod tree;
pub mod types;

pub use config::{
    AlphaErrorOption, BetaErrorOption, HktBuilderConfig, MemoryMode, SecaConfig,
    SecaThresholdConfig, WordImportanceErrorOption,
};
pub use engine::SecaEngine;
pub use error::SecaError;
pub use types::{
    BaselineHktExport, BaselineHktVerboseExport, BaselineNodeExport, BaselineNodeVerboseExport,
    BaselineTreeExport, BaselineTreeVerboseExport, BatchProcessingResult, ClusteringResult,
    EngineSnapshot, SourceBatch, SourceLegendEntry, SourceRecord, UpdateExplanation,
    VerboseSourceRef, VerboseWordRef, WordLegendEntry,
};

pub const ENGINE_VERSION: &str = "0.1.0";
