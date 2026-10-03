//! Where the daemon listens, and whether that needs a token.
//!
//! Loopback TCP is the default and needs no token. Any other address, with
//! `0.0.0.0` included, needs a bearer token, and the daemon refuses to start
//! without one. A Unix socket is kept as an option and is treated like
//! loopback.

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;

/// The daemon's listen address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Listen {
    /// A TCP socket address.
    Tcp(SocketAddr),
    /// A Unix domain socket at the given path.
    Unix(PathBuf),
}

impl Listen {
    /// The default: loopback TCP on port 7720.
    pub const DEFAULT_TCP: &'static str = "127.0.0.1:7720";

    /// Whether only processes on this machine can connect. Loopback means
    /// 127.0.0.0/8 and ::1.
    pub fn is_local(&self) -> bool {
        match self {
            Listen::Tcp(addr) => match addr.ip() {
                IpAddr::V4(ip) => ip.is_loopback(),
                IpAddr::V6(ip) => ip.is_loopback(),
            },
            Listen::Unix(_) => true,
        }
    }
}

impl Default for Listen {
    fn default() -> Self {
        Self::DEFAULT_TCP
            .parse()
            .expect("the default listen address parses")
    }
}

impl FromStr for Listen {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(path) = s.strip_prefix("unix:") {
            if path.is_empty() {
                return Err("a unix socket needs a path after `unix:`".into());
            }
            return Ok(Listen::Unix(PathBuf::from(path)));
        }
        s.parse::<SocketAddr>()
            .map(Listen::Tcp)
            .map_err(|e| format!("expected `host:port` or `unix:/path`: {e}"))
    }
}

impl fmt::Display for Listen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Listen::Tcp(addr) => write!(f, "{addr}"),
            Listen::Unix(path) => write!(f, "unix:{}", path.display()),
        }
    }
}
