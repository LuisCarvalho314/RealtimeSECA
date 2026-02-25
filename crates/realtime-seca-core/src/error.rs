use thiserror::Error;

#[derive(Debug, Error)]
pub enum SecaError {
    #[error("invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("engine state error: {message}")]
    StateError { message: String },

    #[error("serialization error: {message}")]
    SerializationError { message: String },
}
