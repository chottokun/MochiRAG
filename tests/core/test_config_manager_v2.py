import unittest
import os
from unittest.mock import patch
from core.config_manager import ConfigManager, Settings

class TestConfigManagerV2(unittest.TestCase):
    def test_settings_load(self):
        with patch.dict(os.environ, {"SECRET_KEY": "supersecret", "CHROMA_HOST": "localhost"}):
            new_settings = Settings()
            self.assertEqual(new_settings.secret_key, "supersecret")
            self.assertEqual(new_settings.chroma_host, "localhost")

    def test_config_manager_singleton(self):
        cm1 = ConfigManager()
        cm2 = ConfigManager()
        self.assertIs(cm1, cm2)

    def test_get_llm_config(self):
        cm = ConfigManager()
        # Assuming strategies.yaml has a 'main' role
        config = cm.get_llm_config_by_role("main")
        self.assertIsNotNone(config.provider)
        self.assertIsNotNone(config.temperature)

    def test_get_embedding_config(self):
        cm = ConfigManager()
        # Use a model name that exists in strategies.yaml
        config = cm.get_embedding_config("all-MiniLM-L6-v2")
        self.assertIsNotNone(config.provider)

if __name__ == "__main__":
    unittest.main()
