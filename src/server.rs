use std::time::Duration;

use tokio::net::TcpListener;
use tracing::info;

use crate::app::build_router;
use crate::config::RuntimeConfig;
use crate::error::Error;
use crate::proxy::ProxyState;

fn checked_duration_seconds(name: &str, value: f64) -> Result<Duration, Error> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::Config(format!(
            "{name} must be a finite number > 0 (got {value})"
        )));
    }
    Duration::try_from_secs_f64(value)
        .map_err(|_| Error::Config(format!("{name} must be representable as a Duration (got {value})")))
}

/// Poll vLLM `/health` until it responds 200 or the timeout is reached.
///
/// # Errors
///
/// Returns an error if vLLM does not become ready within the configured timeout.
pub async fn wait_vllm_ready(config: &RuntimeConfig) -> Result<(), Error> {
    let base = config.llm_api_base.trim_end_matches('/');
    let url = format!("{base}/health");

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = config.openai_api_key.as_deref() {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {trimmed}"))?,
            );
        }
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .default_headers(headers)
        .build()
        .map_err(Error::HttpClient)?;

    let timeout = checked_duration_seconds("vllm_ready_timeout_s", config.vllm_ready_timeout_s)?;
    let interval = checked_duration_seconds("vllm_ready_interval_s", config.vllm_ready_interval_s)?;
    let start = tokio::time::Instant::now();
    let mut last_notice = Duration::ZERO;

    loop {
        let elapsed = start.elapsed();
        if elapsed > timeout {
            return Err(Error::VllmTimeout {
                url,
                timeout_s: config.vllm_ready_timeout_s,
            });
        }

        match client.get(&url).send().await {
            Ok(resp) if resp.status().as_u16() == 200 => return Ok(()),
            _ => {}
        }

        if elapsed.saturating_sub(last_notice) >= interval {
            last_notice = elapsed;
            info!("waiting for vLLM ({}s elapsed): {url}", elapsed.as_secs());
        }

        tokio::time::sleep(interval).await;
    }
}

async fn serve_gateway(config: RuntimeConfig) -> Result<(), Error> {
    let addr = format!("{}:{}", config.gateway_host, config.gateway_port);
    let state = ProxyState::new(config)?;
    let router = build_router(state);
    let listener = TcpListener::bind(&addr).await?;
    info!("gateway listening on {addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

/// Start the gateway after vLLM becomes ready.
///
/// # Errors
///
/// Returns an error if vLLM readiness polling fails or the server cannot bind.
pub async fn run(config: RuntimeConfig) -> Result<(), Error> {
    wait_vllm_ready(&config).await?;
    info!("vLLM ready: {}", config.llm_api_base);
    serve_gateway(config).await
}

/// Spawn vLLM as a subprocess and run the gateway in the foreground.
///
/// # Errors
///
/// Returns an error if vLLM fails to start or the gateway errors.
pub async fn run_with_vllm(config: RuntimeConfig, vllm_args: Vec<String>) -> Result<(), Error> {
    let mut cmd = tokio::process::Command::new("python");
    cmd.arg("-m").arg("vllm.entrypoints.openai.api_server");
    cmd.args(&vllm_args);

    let mut child = cmd.spawn()?;
    info!("spawned vLLM subprocess (pid {})", child.id().unwrap_or(0));

    let readiness_result = tokio::select! {
        ready = wait_vllm_ready(&config) => ready,
        status = child.wait() => {
            let status = status?;
            Err(Error::VllmProcessExited {
                status: status.to_string(),
            })
        }
    };

    match readiness_result {
        Ok(()) => info!("vLLM ready: {}", config.llm_api_base),
        Err(err) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(err);
        }
    }

    let result = tokio::select! {
        gateway = serve_gateway(config) => gateway,
        status = child.wait() => {
            let status = status?;
            Err(Error::VllmProcessExited {
                status: status.to_string(),
            })
        }
    };

    let _ = child.kill().await;
    let _ = child.wait().await;
    result
}

#[cfg(test)]
mod tests {
    use super::checked_duration_seconds;

    #[test]
    fn checked_duration_rejects_non_positive() {
        assert!(checked_duration_seconds("v", 0.0).is_err());
        assert!(checked_duration_seconds("v", -1.0).is_err());
    }

    #[test]
    fn checked_duration_rejects_nan() {
        assert!(checked_duration_seconds("v", f64::NAN).is_err());
    }

    #[test]
    fn checked_duration_rejects_infinite() {
        assert!(checked_duration_seconds("v", f64::INFINITY).is_err());
    }

    #[test]
    fn checked_duration_rejects_too_large_finite() {
        assert!(checked_duration_seconds("v", 1e50).is_err());
    }

    #[test]
    fn checked_duration_accepts_positive_finite() {
        let duration = checked_duration_seconds("v", 0.25).unwrap();
        assert_eq!(duration.as_millis(), 250);
    }
}
