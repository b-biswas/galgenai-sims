"""Configuration loading utilities for galgenai-sims."""

import yaml
from pathlib import Path


def load_sim_config(config_path=None):
    """
    Load configuration from a YAML file.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to the configuration YAML file.
        If not provided, looks for 'sim_config.yaml' in current directory,
        then falls back to package default config.

    Returns:
    --------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Try default config in package
        package_dir = Path(__file__).parent
        config_path = package_dir.parent.parent / "default_sim_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please provide a config file path or create sim_config.yaml"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
