# Memory runs on bank time, truth on world time

Asphodel uses two clocks. Strength ("does this come to mind?") runs on bank time: full speed for 24 hours after any conversation turn in the bank, and 0.1 speed otherwise. Validity windows, phase, due dates and state confidence ("is this still true?") run on world time. A holiday or an occasionally used bank shouldn't make the assistant forget, but a state like "User is in Berlin this week" really does go stale while nobody is talking.

## Considered Options

- **Calendar time for everything.** A bank used once a month would forget everything between visits.
- **Conversation turns.** An abandoned bank would never fade. A busy day would also make two mentions on the same day count as separate occasions for the lasting floor.
- **One clock shared across banks.** Activity in one bank would age another bank's memories, and banks are meant to be isolated.

## Consequences

- Bank time counts turns only. Ingesting a document doesn't start the clock.
- The three-day spacing for the lasting floor is measured in world time, because whether two mentions were separate occasions is a fact about the world.
- The 0.1 rate for quiet time decides how fast an abandoned bank fades. At 0.1, a memory mentioned once fades out after about 10 weeks at significance 0, 1.7 years at 0.3 and 7 years at 0.5. It's the constant the replay harness is allowed to tune.
- Bank time is computed from when turns happened, not from calendar days, so timezones, daylight saving and travel need no special handling.
