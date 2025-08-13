import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """
    A manager class to handle loading and accessing configuration from a YAML file.
    """
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Path = Path("config/strategies.yaml")):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # In a real scenario, you might want to handle re-instantiation differently
            # but for this singleton-like pattern, we load once.
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: Path):
        """Loads the YAML configuration file."""
        if not config_path.is_file():
            # This is a critical error, the application can't run without its config.
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    @property
    def defaults(self) -> Dict[str, str]:
        """Returns the default settings."""
        return self._config.get("defaults", {})

    @property
    def llms(self) -> Dict[str, Any]:
        """Returns the LLM configurations."""
        return self._config.get("llms", {})

    def get_llm_config(self, name: str = None) -> Dict[str, Any]:
        """
        Returns the configuration for a specific LLM.
        If no name is provided, it returns the default LLM config.
        """
        if name is None:
            name = self.defaults.get("llm")
        if not name:
             raise ValueError("Default LLM name not found in config.")
        config = self.llms.get(name)
        if config is None:
            raise ValueError(f"LLM configuration '{name}' not found.")
        return config

# Create a default instance for easy import across the application
config_manager = ConfigManager()
