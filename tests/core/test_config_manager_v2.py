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

    def test_llm_api_key_override(self):
        # Mock environment variables for API keys
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "GOOGLE_API_KEY": "gemini-test",
            "AZURE_OPENAI_API_KEY": "azure-test"
        }):
            # We need to re-initialize or create a new instance to pick up new settings
            # Since it's a singleton, we might need to reset it for testing
            from core.config_manager import ConfigManager
            ConfigManager._instance = None
            cm = ConfigManager()

            # Check OpenAI
            gpt4_config = cm.get_llm_config_by_role("main") # main is gemma3 in yaml, let's find one that uses openai
            # GPT-4o is a provider in strategies.yaml
            gpt4o_config = cm.config.llms.providers["gpt-4o"]
            self.assertEqual(gpt4o_config.api_key, "sk-test")

            # Check Gemini
            gemini_config = cm.config.llms.providers["gemini-1.5-pro"]
            self.assertEqual(gemini_config.api_key, "gemini-test")

if __name__ == "__main__":
    unittest.main()
