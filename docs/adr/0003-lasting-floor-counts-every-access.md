# The lasting floor counts every access

Strength is the significance boost plus the larger of recent use and a lasting floor. The floor counts every access a memory has ever had, taking at most one per three calendar days, and it never goes down. When a memory's validity window closes, only recent use starts over. It restarts from a single recency boost, placed when the end became known, plus any accesses after the window ended. Excluding accesses inside a closed window altogether would make the floor collapse when a window closed. Ended memories such as "User lived in Berlin" would fade and could then be purged, contradicting "an ended memory stays in history".

## Considered Options

- **Exclude accesses inside a closed window altogether**. This stops an upcoming appointment from being inflated by anticipation, but it also erases months of genuine use of a state or task.
- **Count them as normal.** This lets a much-discussed appointment stay strong long after it has passed.
- **A rule per kind** (drop for events, keep for states). We rejected it because extraction assigns the kind and can get it wrong, so a memory's lifetime would depend on which kind extraction picked.

## Consequences

- The three-day spacing is what separates anticipation from knowledge. An appointment mentioned daily for a week counts as about 3 separate occasions and fades. A state used every ten days for three months counts as 10 and stays in history.
- A replacement memory inherits its predecessor's accesses when the predecessor was retracted or refined, but never when it was ended.
- The recency boost isn't an access. It counts towards recent use only, and never as an occasion for the floor.
- A window closes at the end of its `valid_until`'s unit, in the source's timezone. An event with no `valid_until` is a point event, and its window is `valid_from`'s unit, so it closes when that unit ends. Other kinds have no end until one is stated or they're ended.
