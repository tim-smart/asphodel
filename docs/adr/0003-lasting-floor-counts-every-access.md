# The lasting floor counts every access

Strength is the significance boost plus the larger of recent use and a lasting floor. The floor counts every access a memory has ever had, taking at most one per three calendar days, and it never goes down. When a memory's validity window closes, only recent use starts over. It restarts from a single recency boost, placed when the end became known, plus any accesses after the window ended. This amends "Should memories expire, and how?" (TIM-85), which excluded accesses inside a closed window altogether. That rule made the floor collapse when a window closed. It also meant ended memories such as "User lived in Berlin" faded and could then be purged, contradicting "an ended memory stays in history".

## Considered Options

- **Exclude accesses inside a closed window altogether** (TIM-85). This stops an upcoming appointment from being inflated by anticipation, but it also erases months of genuine use of a state or task.
- **Count them as normal.** This lets a much-discussed appointment stay strong long after it has passed.
- **A rule per kind** (drop for events, keep for states). We rejected it because extraction assigns the kind and can get it wrong, so a memory's lifetime would depend on which kind extraction picked.

## Consequences

- The three-day spacing is what separates anticipation from knowledge. An appointment mentioned daily for a week counts as about 3 separate occasions and fades. A state used every ten days for three months counts as 10 and stays in history.
- A replacement memory inherits its predecessor's accesses when the predecessor was retracted or refined, but never when it was ended.

Decided in "Strength model: decay, reinforcement and significance" (TIM-91) on 2026-09-30.
