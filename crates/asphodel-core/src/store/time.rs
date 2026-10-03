//! How times are stored: an INTEGER of microseconds since the Unix epoch,
//! UTC. Integers compare and index cheaply, and microseconds cover
//! ±292,000 years, which is more than jiff's own range.

use jiff::Timestamp;

/// The database form of a timestamp.
pub fn micros(at: Timestamp) -> i64 {
    at.as_microsecond()
}

/// A timestamp read back from the database. Panics on a value outside
/// jiff's range, which no code path writes.
pub fn timestamp(micros: i64) -> Timestamp {
    Timestamp::from_microsecond(micros).expect("a stored timestamp is within range")
}
