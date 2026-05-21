"""Threshold config store."""

import pytest


def test_config_returns_defaults_when_empty(temp_data_dir):
    import os
    from config_store import ConfigStore, DEFAULT_CONFIG
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    cur = cfg.get_all()
    for k, v in DEFAULT_CONFIG.items():
        assert cur[k] == v


def test_set_and_get_persist(temp_data_dir):
    import os
    from config_store import ConfigStore
    db_path = os.path.join(temp_data_dir, "test.db")
    cfg = ConfigStore(db_path)
    cfg.set("circular_amount_tolerance", 0.30)
    cfg2 = ConfigStore(db_path)
    assert cfg2.get("circular_amount_tolerance") == 0.30


def test_unknown_key_rejected(temp_data_dir):
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    with pytest.raises(ValueError):
        cfg.set_many({"this_is_not_a_real_setting": 42})
