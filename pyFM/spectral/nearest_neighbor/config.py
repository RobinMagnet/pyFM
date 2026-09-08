"""Tunable parameters for nearest-neighbour backend selection.
Defaults are loosely measured.
"""

DEFAULTS = {
    # Dimension p at or below which a kd-tree is useful
    "kdtree_max_dim": 8,
    # Below this many reference points n1, bruteforce is faster than kdtree
    "tree_min_points": 1000,
    # KDTree leafsize (scipy implementation)
    "leaf_size": 40,
    # scikit-learn working_memory (MB), to avoid oom on large brute-force queries
    "working_memory_mb": 1024,
    # n1 * n2 above which parallel tree queries helps
    "parallel_min_work": 1e8,
}

# This allows to modify config while keeping the defaults intact.
_config = dict(DEFAULTS)


def get_config(key=None):
    """Return the current configuration, or value of specific key

    Parameters
    ----------
    key : str, optional
        One of the keys of :data:`DEFAULTS`. If None, the whole configuration.

    Returns
    -------
    config : dict or object
        A copy of the configuration, or the single value bound to key.
    """
    if key is None:
        return dict(_config)
    if key not in _config:
        raise KeyError(f"unknown config key {key!r}; valid keys are {sorted(_config)}")
    return _config[key]


def set_config(**kwargs):
    """Override one or more thresholds. Keys are those of :data:`DEFAULTS`.

    Returns
    -------
    config : dict
        A copy of the configuration after the update.
    """
    unknown = set(kwargs) - set(_config)
    if unknown:
        raise KeyError(
            f"unknown config key(s) {sorted(unknown)}; valid keys are {sorted(_config.keys())}"
        )
    _config.update(kwargs)
    return dict(_config)


def reset_config():
    """Restore every threshold to its default.

    Returns
    -------
    config : dict
        A copy of the restored configuration.
    """
    _config.clear()
    _config.update(DEFAULTS)
    return dict(_config)
