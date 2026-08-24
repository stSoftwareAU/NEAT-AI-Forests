//! NEAT-AI-Forests — experimental residual decision-tree optimiser for
//! already-fit NEAT-AI creatures.
//!
//! > **Find all the dirty tricks that uncover real improvements — but trust
//! > only the scorer.**
//!
//! The crate is organised as a pipeline. Each module owns one stage and is
//! deliberately independent of the acceptance decision, which lives solely in
//! [`promote`] and is delegated to the authoritative NEAT-AI-scorer:
//!
//! | Stage | Module | Issue |
//! |---|---|---|
//! | immutable incumbent + checksum | [`incumbent`] | #2 |
//! | authoritative baseline + parity | [`baseline`], [`scorer`] | #2 |
//! | quantile-bin cache | [`bins`] | #3 |
//! | residual extraction | [`residuals`] | #4 |
//! | CPU histogram stump search | [`histogram`] | #5 |
//! | portable patch + `IF` graft | [`patch`], [`graft`] | #7 |
//! | candidate population | [`candidates`] | #8 |
//! | screen + promote | [`promote`] | #9 |
//! | evolution loop + journal | [`run`], [`journal`] | #10 |
//! | depth-2/3 trees | [`tree`] | #11 |
//! | sampling / jitter / diversity | [`strategies`] | #12 |
//! | XGBoost control | [`xgboost`] | #13 |
//! | oblique splits | [`oblique`] | #14 |
//! | economics report | [`report`] | #15 |
//!
//! Cheap search (histograms, samples, random guesses) only ever *proposes*
//! candidates. Only a full-corpus NEAT-AI-scorer result can *accept* one.

#![warn(missing_docs)]

pub mod baseline;
pub mod bins;
pub mod cancel;
pub mod candidates;
pub mod config;
pub mod corpus;
pub mod graft;
pub mod histogram;
pub mod incumbent;
pub mod journal;
pub mod learnings;
pub mod log;
pub mod meta;
pub mod oblique;
pub mod patch;
pub mod promote;
pub mod report;
pub mod residuals;
pub mod run;
pub mod scorer;
pub mod strategies;
pub mod tools;
pub mod tree;
pub mod xgboost;

pub use baseline::{AuthoritativeBaseline, ParityPolicy, establish_baseline};
pub use bins::{BinCache, BinCacheError, DEFAULT_BINS, ensure_bin_cache};
pub use cancel::CancelToken;
pub use candidates::{Candidate, CandidateConfig, generate_candidates};
pub use config::{ForestsConfig, GraftConstants, GrowthPolicy};
pub use corpus::{CorpusInfo, corpus_info};
pub use graft::{GraftError, graft_patch, graft_patch_with};
pub use histogram::{
    BinnedChunk, ChunkSource, HistogramSet, SearchControls, StumpCandidate, StumpKind,
    search_stumps,
};
pub use incumbent::{Incumbent, IncumbentError, load_incumbent};
pub use journal::{JournalLine, append_journal_line, read_journal};
pub use patch::{Condition, Node, Patch, Provenance, Term};
pub use promote::{PromoteConfig, PromotionOutcome, ScreenOutcome, screen_and_promote};
pub use report::{JournalReport, report_from_journal};
pub use residuals::{ResidualCache, ResidualStats, ensure_residual_cache};
pub use run::{RunResult, StopReason, run_forests};
pub use scorer::{DirectoryScorer, ExternalScorer, ScoreResult, ScorerError, ScorerMode};
pub use tree::{TreeSearchControls, grow_tree};
