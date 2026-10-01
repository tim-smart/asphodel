# Upgrading

## Schema version 2: send each bank's config again

Schema version 2 adds `speaker_ids`, the only mapping a turn's speaker is
resolved through ("API surface and Hermes transport", TIM-94, decision 1).
The migration from version 1 fails closed. It carries over the speakers
ingest created, whose platform ids version 1 recorded, but **not the owner's
platform ids**. Version 1 kept those only as aliases of the `user` entity,
next to the owner's current and former names, so a former name that looks
like a platform id can't be told from a real id, and guessing could make a
stranger the owner.

Until a bank's config is sent again, a turn from the owner's platform
account is attributed to a stranger entity, not to `user`. Sending the
config maps the owner's ids to `user`, taking them over from any entity
that held them, and logs a `speaker_id_set` edit.

After upgrading the daemon, send the config of every bank that has owner
platform ids, before Hermes ingests more turns:

- Restart Hermes (or each gateway and CLI process). The plugin's
  `initialize` calls `PUT /v1/banks/{bank}` with the owner's platform ids
  from its config. Plugin instances that were already running don't call it
  again on their own.
- Or call `PUT /v1/banks/{bank}` with `owner_platform_ids` yourself, or
  `asphodel bank config`, once the HTTP API and CLI land (TIM-110).

The daemon logs a warning when it migrates a store from version 1, as a
reminder.
