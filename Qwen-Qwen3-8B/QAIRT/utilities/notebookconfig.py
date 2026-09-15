# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config_to_env(config_json_filename):
    """
    Load configuration from JSON file and inject into os.environ.

    Args:
        config_json_filename: Name of the JSON config file
    """
    # Get the current working directory (notebook's directory in Jupyter)
    notebook_dir = os.getcwd()

    # The config file should be in a 'config' subdirectory relative to notebook
    # or in the same directory as this module
    config_path = None

    # Try to find config file in multiple locations
    search_paths = [
        os.path.join(notebook_dir, config_json_filename),
        os.path.join(notebook_dir, 'config', config_json_filename),
    ]

    for path in search_paths:
        if os.path.exists(path):
            config_path = path
            break

    if config_path is None:
        config_path = os.path.join(notebook_dir, config_json_filename)

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)

        # Inject each config value into environment
        for key, value in config_data.items():
            env_key = key.upper()
            os.environ[env_key] = str(value)
            logger.info(f"Set {env_key}={value}")

        logger.info(f"Successfully loaded config from {config_json_filename}")
        logger.info(f"Config loaded from: {config_path}")

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        logger.error(f"Searched in: {search_paths}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def get_config(config_json_filename):
    """
    Legacy function for backward compatibility.
    Now injects config into os.environ and sets working directory to notebook location.

    Args:
        config_json_filename: Name of the JSON config file
    """
    load_config_to_env(config_json_filename)
