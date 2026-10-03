"""The dashboard's view of the plugin config.

Hermes loads this file by path and never imports the package, so it
may import only ``plugins.memory.config_schema``. Keep it in step with
``config.config_schema()``."""

from __future__ import annotations

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_SECRET,
    KIND_TEXT,
    STORAGE_FLAT_JSON,
    ProviderConfigSchema,
    ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="asphodel",
    label="Asphodel",
    storage=STORAGE_FLAT_JSON,
    fields=(
        ProviderField(
            key="url",
            label="Daemon URL",
            kind=KIND_TEXT,
            default="http://127.0.0.1:7720",
            description="http://host:port or unix:/path. ASPHODEL_URL in the environment overrides it.",
            inline=True,
        ),
        ProviderField(
            key="token",
            label="Bearer token",
            kind=KIND_SECRET,
            env_key="ASPHODEL_TOKEN",
            description="Required only when the daemon listens off loopback.",
        ),
        ProviderField(
            key="bank",
            label="Bank",
            kind=KIND_TEXT,
            description="One bank per Hermes profile. Defaults to the profile name.",
            inline=True,
        ),
        ProviderField(key="owner_name", label="Owner's name", kind=KIND_TEXT),
        ProviderField(
            key="owner_platform_ids",
            label="Owner's platform ids",
            kind=KIND_TEXT,
            description="Comma-separated speaker ids such as discord:1234.",
        ),
        ProviderField(
            key="assistant_name",
            label="Assistant's name",
            kind=KIND_TEXT,
            description="Defaults to the profile name.",
        ),
        ProviderField(
            key="timezone",
            label="Default timezone",
            kind=KIND_TEXT,
            description="IANA name. Defaults to Hermes' configured timezone.",
        ),
        ProviderField(
            key="ingest",
            label="Ingest turns",
            kind=KIND_BOOL,
            default="true",
            description="Off means recall and injection only; nothing new is remembered.",
        ),
    ),
)
