"""Setup: ``save_config`` into
``config.json``, and ``post_setup`` turning Hermes' built-in memory off and
activating the provider."""

import json

from conftest import plugin


def test_save_config_writes_non_secret_values(hermes_home):
    provider = plugin.AsphodelMemoryProvider()
    provider.save_config({"url": "http://a:1", "bank": "tim", "owner_platform_ids": "discord:111, telegram:222", "token": "x"}, str(hermes_home))
    saved = json.loads((hermes_home / "asphodel" / "config.json").read_text())
    assert saved["url"] == "http://a:1"
    assert saved["bank"] == "tim"
    assert saved["owner_platform_ids"] == ["discord:111", "telegram:222"]
    assert "token" not in saved


def test_post_setup_turns_builtin_memory_off_and_activates_the_provider(hermes_home, hermes):
    provider = plugin.AsphodelMemoryProvider()
    config = {"memory": {"memory_enabled": True, "user_profile_enabled": True, "provider": "hindsight"}}
    # The schema's fields, then no to the history backfill.
    answers = iter(["http://10.0.0.5:7720", "", "Tim", "discord:111", "", "", "", "n"])
    provider.post_setup(str(hermes_home), config, prompt=lambda label: next(answers))
    assert config["memory"]["memory_enabled"] is False
    assert config["memory"]["user_profile_enabled"] is False
    assert config["memory"]["provider"] == "asphodel"
    assert hermes.saved == [config]
    saved = json.loads((hermes_home / "asphodel" / "config.json").read_text())
    assert saved["url"] == "http://10.0.0.5:7720"
    assert saved["owner_name"] == "Tim"
    assert saved["owner_platform_ids"] == ["discord:111"]


def test_post_setup_does_not_prompt_for_the_secret(hermes_home):
    labels = []

    def prompt(label):
        labels.append(label)
        return ""

    plugin.AsphodelMemoryProvider().post_setup(str(hermes_home), {}, prompt=prompt)
    assert not any("token" in label.lower() for label in labels)
