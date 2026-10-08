//! The deletion fingerprint.
//!
//! Purge and the source sweep are pure functions of the access log, the
//! clock and the values below, so changing one of them on a live store can
//! delete things on the next sweep. The store keeps this hash, and purge and
//! the sweep pause when it changes until an operator acknowledges it. The
//! hash covers values, not the git SHA, so a deploy that doesn't change them
//! doesn't pause purge.

use std::fmt;

use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::config::Tuning;
use crate::constants::{self, Significance};

/// Every value that decides an irreversible deletion. The store keeps them
/// next to their fingerprint, so `purge plan` can say which changed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeletionInputs {
    pub s: f64,
    pub tau: f64,
    pub a: f64,
    pub c: f64,
    pub d_max: f64,
    pub g: f64,
    pub n0: f64,
    pub min_access_age_days: f64,
    pub floor_spacing_days: f64,
    /// Inputs recorded before it existed read as 1.0, the margin purge
    /// already had then through δ's default.
    #[serde(default = "default_never_purged_margin")]
    pub never_purged_margin: f64,
    pub weight_created: f64,
    /// `strength.access_weights.used`, under the name it had when it was a
    /// constant, so its default keeps every stored fingerprint.
    pub weight_used: f64,
    pub weight_mentioned_again: f64,
    pub weight_confirmed: f64,
    pub weight_window_close: f64,
    pub full_speed_window_secs: u64,
    /// The tuned values of [`Significance::ALL`], lowest first.
    pub significance: [f64; 5],
    pub significance_kept: f64,
    /// `strength.corroborate_used`. Inputs recorded before it existed read
    /// as off.
    #[serde(default)]
    pub corroborate_used: bool,
    pub quiet_rate: f64,
    pub delta: Option<f64>,
    pub overdue_days: u32,
    pub source_horizon_days: u32,
}

fn default_never_purged_margin() -> f64 {
    1.0
}

impl DeletionInputs {
    /// The fixed strength constants of this build, with the deletion inputs
    /// from `tuning`.
    pub fn new(tuning: &Tuning) -> Self {
        Self {
            s: constants::S,
            tau: constants::TAU,
            a: constants::A,
            c: constants::C,
            d_max: constants::D_MAX,
            g: constants::G,
            n0: constants::N0,
            min_access_age_days: constants::MIN_ACCESS_AGE_DAYS,
            floor_spacing_days: constants::FLOOR_SPACING_DAYS,
            never_purged_margin: constants::NEVER_PURGED_MARGIN,
            weight_created: constants::WEIGHT_CREATED,
            weight_used: tuning.strength.access_weights.used,
            weight_mentioned_again: constants::WEIGHT_MENTIONED_AGAIN,
            weight_confirmed: constants::WEIGHT_CONFIRMED,
            weight_window_close: constants::WEIGHT_WINDOW_CLOSE,
            full_speed_window_secs: constants::FULL_SPEED_WINDOW.as_secs(),
            significance: tuning.strength.significance.values(),
            significance_kept: constants::SIGNIFICANCE_KEPT,
            corroborate_used: tuning.strength.corroborate_used,
            quiet_rate: tuning.clock.quiet_rate,
            delta: tuning.purge.delta,
            overdue_days: tuning.agenda.overdue_days,
            source_horizon_days: tuning.purge.source_horizon_days,
        }
    }
}

impl DeletionInputs {
    /// The values that differ from `stored`, by tuning key, with
    /// `constants` standing for every strength constant fixed in code.
    pub fn changed_from(&self, stored: &DeletionInputs) -> Vec<String> {
        let mut changed = Vec::new();
        let constants = |inputs: &DeletionInputs| DeletionInputs {
            weight_used: 0.0,
            significance: [0.0; 5],
            corroborate_used: false,
            quiet_rate: 0.0,
            delta: None,
            overdue_days: 0,
            source_horizon_days: 0,
            ..inputs.clone()
        };
        if constants(self) != constants(stored) {
            changed.push("constants".to_string());
        }
        if self.weight_used != stored.weight_used {
            changed.push("strength.access_weights.used".to_string());
        }
        if self.significance != stored.significance {
            changed.push("strength.significance".to_string());
        }
        if self.corroborate_used != stored.corroborate_used {
            changed.push("strength.corroborate_used".to_string());
        }
        if self.quiet_rate != stored.quiet_rate {
            changed.push("clock.quiet_rate".to_string());
        }
        if self.delta != stored.delta {
            changed.push("purge.delta".to_string());
        }
        if self.overdue_days != stored.overdue_days {
            changed.push("agenda.overdue_days".to_string());
        }
        if self.source_horizon_days != stored.source_horizon_days {
            changed.push("purge.source_horizon_days".to_string());
        }
        changed
    }
}

/// A SHA-256 over the deletion inputs, shown as lowercase hex. It's what
/// `asphodel purge ack --hash` quotes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fingerprint(String);

impl Fingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A fingerprint read back from the store, as it was recorded.
    pub(crate) fn from_stored(hex: String) -> Self {
        Self(hex)
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for Fingerprint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

/// Hashes the deletion inputs. It's a pure function: the same values give
/// the same fingerprint on every build and platform.
///
/// Each value goes in under its name, so two values can't swap without
/// changing the hash. Floats go in as their IEEE 754 bits, with −0 folded
/// into 0. Renaming or adding a field changes every stored fingerprint and
/// pauses purge once, which is the safe direction. A switch added later goes
/// in only when it's on, so its default keeps every stored fingerprint.
pub fn deletion_fingerprint(inputs: &DeletionInputs) -> Fingerprint {
    let mut hash = Hasher(Sha256::new());
    hash.0.update(b"asphodel deletion fingerprint v1\n");

    let DeletionInputs {
        s,
        tau,
        a,
        c,
        d_max,
        g,
        n0,
        min_access_age_days,
        floor_spacing_days,
        never_purged_margin,
        weight_created,
        weight_used,
        weight_mentioned_again,
        weight_confirmed,
        weight_window_close,
        full_speed_window_secs,
        significance,
        significance_kept,
        corroborate_used,
        quiet_rate,
        delta,
        overdue_days,
        source_horizon_days,
    } = inputs;

    hash.float("s", *s);
    hash.float("tau", *tau);
    hash.float("a", *a);
    hash.float("c", *c);
    hash.float("d_max", *d_max);
    hash.float("g", *g);
    hash.float("n0", *n0);
    hash.float("min_access_age_days", *min_access_age_days);
    hash.float("floor_spacing_days", *floor_spacing_days);
    hash.float("never_purged_margin", *never_purged_margin);
    hash.float("weight.created", *weight_created);
    hash.float("weight.used", *weight_used);
    hash.float("weight.mentioned_again", *weight_mentioned_again);
    hash.float("weight.confirmed", *weight_confirmed);
    hash.float("weight.window_close", *weight_window_close);
    hash.int("full_speed_window_secs", *full_speed_window_secs);
    for (level, value) in Significance::ALL.iter().zip(significance) {
        hash.float(&format!("significance.{level:?}"), *value);
    }
    hash.float("significance.kept", *significance_kept);
    if *corroborate_used {
        hash.field("strength.corroborate_used", b"on");
    }
    hash.float("clock.quiet_rate", *quiet_rate);
    match delta {
        Some(delta) => hash.float("purge.delta", *delta),
        None => hash.field("purge.delta", b"never"),
    }
    hash.int("agenda.overdue_days", u64::from(*overdue_days));
    hash.int("purge.source_horizon_days", u64::from(*source_horizon_days));

    let digest = hash.0.finalize();
    Fingerprint(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

struct Hasher(Sha256);

impl Hasher {
    fn field(&mut self, name: &str, value: &[u8]) {
        self.0.update(name.as_bytes());
        self.0.update(b"=");
        self.0.update(value);
        self.0.update(b"\n");
    }

    fn float(&mut self, name: &str, value: f64) {
        // Adding 0.0 folds −0 into 0 and leaves every other value alone.
        let mut bytes = [b'f'; 9];
        bytes[1..].copy_from_slice(&(value + 0.0).to_bits().to_le_bytes());
        self.field(name, &bytes);
    }

    fn int(&mut self, name: &str, value: u64) {
        let mut bytes = [b'i'; 9];
        bytes[1..].copy_from_slice(&value.to_le_bytes());
        self.field(name, &bytes);
    }
}

impl Tuning {
    /// The deletion fingerprint of this tuning under this build's constants.
    pub fn deletion_fingerprint(&self) -> Fingerprint {
        deletion_fingerprint(&DeletionInputs::new(self))
    }
}
