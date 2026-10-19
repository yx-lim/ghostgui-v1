"""Legacy workspace payload defaults and lossless detached merging."""

from copy import deepcopy


def default_workspace_payload(document):
    return {
        "schema_version": 1, "current_time": float(document.current_time),
        "timeline_duration": float(document.timeline_duration),
        "active_index": int(document.active_index),
        "display": {"export_dt": 0.01, "dsms_motion_speed": 1.0},
    }


def merge_workspace_payload(existing, updates):
    """Retain unknown fields, including nested UI/provider extension values."""
    result = deepcopy(dict(existing or {}))
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_workspace_payload(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
