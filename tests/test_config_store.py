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


def test_non_numeric_value_rejected(temp_data_dir):
    """A numeric threshold must reject a string value (else detectors TypeError)."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    with pytest.raises(ValueError):
        cfg.set_many({"circular_amount_tolerance": "not_a_number"})
    # Nothing should have been persisted
    assert isinstance(cfg.get("circular_amount_tolerance"), (int, float))


def test_negative_value_rejected(temp_data_dir):
    """Thresholds are all non-negative; a negative value is invalid."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    with pytest.raises(ValueError):
        cfg.set_many({"circular_amount_tolerance": -50})


def test_bool_value_rejected(temp_data_dir):
    """True/False must not slip through as 1/0 into a numeric threshold."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    with pytest.raises(ValueError):
        cfg.set_many({"circular_max_cycle_length": True})


def test_partial_update_is_atomic(temp_data_dir):
    """If any value in a batch is invalid, none are persisted."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    before = cfg.get("smurfing_threshold")
    with pytest.raises(ValueError):
        cfg.set_many({"smurfing_threshold": 999999, "circular_amount_tolerance": -1})
    assert cfg.get("smurfing_threshold") == before


def test_infinity_rejected(temp_data_dir):
    """Infinity is a float but not a usable threshold; it 500s on serialize."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    with pytest.raises(ValueError):
        cfg.set_many({"circular_amount_tolerance": float("inf")})


def test_nan_rejected(temp_data_dir):
    """NaN is the dangerous one: every `x <= NaN` is False, silently disabling
    a detector with no error. Must be rejected outright."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    with pytest.raises(ValueError):
        cfg.set_many({"circular_amount_tolerance": float("nan")})


def test_valid_numeric_update_still_works(temp_data_dir):
    """Valid in-range updates persist (int key accepts whole-number float)."""
    import os
    from config_store import ConfigStore
    cfg = ConfigStore(os.path.join(temp_data_dir, "test.db"))
    cfg.set_many({"circular_amount_tolerance": 0.25, "circular_max_cycle_length": 6.0})
    assert cfg.get("circular_amount_tolerance") == 0.25
    assert cfg.get("circular_max_cycle_length") == 6
    assert isinstance(cfg.get("circular_max_cycle_length"), int)
