use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to build HTTP client")]
    HttpClient(#[source] reqwest::Error),

    #[error("vLLM not ready within {timeout_s:.0}s at {url}")]
    VllmTimeout { url: String, timeout_s: f64 },

    #[error("vLLM subprocess exited before becoming ready: {status}")]
    VllmProcessExited { status: String },

    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("invalid header value")]
    InvalidHeader(#[from] reqwest::header::InvalidHeaderValue),

    #[error("{0}")]
    Config(String),
}
