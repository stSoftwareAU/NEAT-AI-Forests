//! Zero-dependency coloured stderr logging.
//!
//! Everything goes to **stderr** so stdout stays clean for machine-readable
//! output such as `neat_ai_forests report`. Colour is enabled only when stderr
//! is a terminal and neither `NO_COLOR` nor `FORESTS_NO_COLOR` is set.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU8, Ordering};

static COLOUR_ENABLED: AtomicU8 = AtomicU8::new(2); // 0=off 1=on 2=unset

fn colour_enabled() -> bool {
    match COLOUR_ENABLED.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let enabled = std::io::stderr().is_terminal()
                && std::env::var_os("NO_COLOR").is_none()
                && std::env::var_os("FORESTS_NO_COLOR").is_none();
            COLOUR_ENABLED.store(u8::from(enabled), Ordering::Relaxed);
            enabled
        }
    }
}

fn paint(code: &str, text: &str) -> String {
    if colour_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

/// Informational line (`●`).
pub fn info(msg: &str) {
    eprintln!("{} {msg}", paint("1;36", "●"));
}

/// Success line (`✓`).
pub fn ok(msg: &str) {
    eprintln!("{} {msg}", paint("1;32", "✓"));
}

/// Warning line (`⚠`).
pub fn warn(msg: &str) {
    eprintln!("{} {msg}", paint("1;33", "⚠"));
}

/// Indented, dimmed detail line.
pub fn detail(msg: &str) {
    eprintln!("  {}", paint("2", msg));
}

/// Flush stderr (used before spawning the scorer so interleaving is tidy).
pub fn flush() {
    let _ = std::io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_is_plain_when_colour_disabled() {
        COLOUR_ENABLED.store(0, Ordering::Relaxed);
        assert_eq!(paint("1", "x"), "x");
    }
}
