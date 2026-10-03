//! Supersession chains: which accesses a memory inherits and what a purge or
//! forget takes.
//!
//! A chain is every memory connected along `superseded_by`. It's a tree,
//! because two memories can be refined into one, and its head is the one
//! memory with no `superseded_by`. `ended_by` is never a chain link: "moved
//! to Lisbon" doesn't inherit "lives in Berlin".

use std::collections::{BTreeSet, HashMap};

/// One memory's links, by rowid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Link {
    pub id: i64,
    pub superseded_by: Option<i64>,
    pub ended_by: Option<i64>,
}

/// The head of the chain `id` is in. Follows `superseded_by` until it runs
/// out, or stops at the first memory seen twice should the links ever form
/// a cycle.
pub fn chain_head(links: &[Link], id: i64) -> i64 {
    let next: HashMap<i64, i64> = links
        .iter()
        .filter_map(|l| l.superseded_by.map(|by| (l.id, by)))
        .collect();
    let mut seen = BTreeSet::from([id]);
    let mut head = id;
    while let Some(&by) = next.get(&head) {
        if !seen.insert(by) {
            break;
        }
        head = by;
    }
    head
}

/// A bank's chains, indexed once for many head lookups. Each lookup walks
/// only as far as a memory whose head is already known, and remembers the
/// head for every memory it passed, so looking up every version of a long
/// chain costs about as much as walking it once. Heads are as
/// [`chain_head`] finds them.
#[derive(Debug, Clone, Default)]
pub struct Chains {
    next: HashMap<i64, i64>,
    heads: HashMap<i64, i64>,
}

impl Chains {
    pub fn new(links: &[Link]) -> Self {
        Self {
            next: links
                .iter()
                .filter_map(|l| l.superseded_by.map(|by| (l.id, by)))
                .collect(),
            heads: HashMap::new(),
        }
    }

    /// The head of the chain `id` is in.
    pub fn head(&mut self, id: i64) -> i64 {
        if let Some(&head) = self.heads.get(&id) {
            return head;
        }
        let mut path = vec![id];
        let mut seen = BTreeSet::from([id]);
        let mut head = id;
        while let Some(&by) = self.next.get(&head) {
            if let Some(&known) = self.heads.get(&by) {
                head = known;
                break;
            }
            if !seen.insert(by) {
                break;
            }
            head = by;
            path.push(by);
        }
        for member in path {
            self.heads.insert(member, head);
        }
        head
    }
}

/// Every memory in the chain `id` is in, from any member.
pub fn chain(links: &[Link], id: i64) -> BTreeSet<i64> {
    inherits_from(links, chain_head(links, id))
}

/// `head` and every memory that reaches it along `superseded_by`: the
/// memories whose accesses `head`'s strength counts.
pub fn inherits_from(links: &[Link], head: i64) -> BTreeSet<i64> {
    let mut predecessors: HashMap<i64, Vec<i64>> = HashMap::new();
    for link in links {
        if let Some(by) = link.superseded_by {
            predecessors.entry(by).or_default().push(link.id);
        }
    }
    let mut found = BTreeSet::from([head]);
    let mut pending = vec![head];
    while let Some(id) = pending.pop() {
        for &earlier in predecessors.get(&id).into_iter().flatten() {
            if found.insert(earlier) {
                pending.push(earlier);
            }
        }
    }
    found
}
