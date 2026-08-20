//! Cooperative cancellation (SIGINT/SIGTERM).
//!
//! The signal handler only sets a flag; the evolution loop polls it between
//! stages so a run always finishes writing a valid journal and `best.json`.
//! A second signal force-quits with exit code 130.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared cancellation flag.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// New, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Underlying flag (for signal registration).
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag)
    }
}

/// Exit code used when a second signal force-quits the process (128 + SIGINT).
#[cfg(unix)]
pub const FORCE_QUIT_EXIT_CODE: i32 = 130;

/// Install SIGINT/SIGTERM handlers that set the token on the first signal and
/// force-quit on the second.
#[cfg(unix)]
pub fn install_cancel_signals(token: &CancelToken) -> Result<(), String> {
    use signal_hook::consts::{SIGINT, SIGTERM};
    for signal in [SIGINT, SIGTERM] {
        // Registered first so it runs first: it only fires once the flag is
        // already set, i.e. on the second signal.
        signal_hook::flag::register_conditional_shutdown(
            signal,
            FORCE_QUIT_EXIT_CODE,
            token.flag(),
        )
        .map_err(|e| format!("cannot install force-quit handler for signal {signal}: {e}"))?;
        signal_hook::flag::register(signal, token.flag())
            .map_err(|e| format!("cannot install cancellation handler for signal {signal}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_flag() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        assert!(b.is_cancelled());
    }
}
