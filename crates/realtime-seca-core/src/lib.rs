pub mod config;
pub mod engine;
pub mod error;
pub mod tree;
pub mod types;

pub use config::{HktBuilderConfig, MemoryMode, SecaConfig, SecaThresholdConfig};
pub use engine::SecaEngine;
pub use error::SecaError;
pub use types::{
    BaselineHktExport, BaselineHktVerboseExport, BaselineNodeExport, BaselineNodeVerboseExport,
    BaselineTreeExport, BaselineTreeVerboseExport, BatchProcessingResult, EngineSnapshot,
    HktDecisionDiagnostics, HktDisplayDiagnostics, SourceBatch, SourceLegendEntry, SourceRecord,
    UpdateExplanation, VerboseSourceRef, VerboseWordRef, WordLegendEntry,
};

pub const ENGINE_VERSION: &str = "0.2.0-seca-light";
