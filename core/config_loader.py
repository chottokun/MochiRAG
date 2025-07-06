import yaml
from pathlib import Path
from typing import Dict, Any, List

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "strategies.yaml"

class StrategyConfigError(Exception):
    """Custom exception for strategy configuration errors."""
    pass

def load_strategy_config() -> Dict[str, Any]:
    """Loads the strategy configuration from the YAML file."""
    if not CONFIG_FILE_PATH.exists():
        raise StrategyConfigError(f"Configuration file not found at {CONFIG_FILE_PATH}")
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            raise StrategyConfigError("Configuration file is empty or invalid.")
        return config
    except yaml.YAMLError as e:
        raise StrategyConfigError(f"Error parsing YAML configuration file: {e}") from e
    except Exception as e:
        raise StrategyConfigError(f"An unexpected error occurred while loading configuration: {e}") from e

# --- Example Usage (for testing the loader itself) ---
if __name__ == '__main__':
    try:
        config_data = load_strategy_config()
        print("Successfully loaded strategy configuration:")
        # print(yaml.dump(config_data, indent=2)) # Pretty print YAML

        # Accessing specific parts for verification
        print("\n--- Embedding Strategies ---")
        default_embedding = config_data.get('embedding_strategies', {}).get('default')
        available_embeddings = config_data.get('embedding_strategies', {}).get('available', [])
        print(f"Default Embedding: {default_embedding}")
        for emb_strat in available_embeddings:
            print(f"  - Name: {emb_strat.get('name')}, Type: {emb_strat.get('type')}, Model: {emb_strat.get('model_name')}")

        print("\n--- Chunking Strategies ---")
        default_chunking = config_data.get('chunking_strategies', {}).get('default')
        available_chunking = config_data.get('chunking_strategies', {}).get('available', [])
        print(f"Default Chunking: {default_chunking}")
        for chk_strat in available_chunking:
            print(f"  - Name: {chk_strat.get('name')}, Type: {chk_strat.get('type')}, Params: {chk_strat.get('params')}")
            if chk_strat.get('embedding_strategy_ref'):
                print(f"    Refers to embedding: {chk_strat.get('embedding_strategy_ref')}")


        print("\n--- RAG Search Strategies ---")
        default_rag_search = config_data.get('rag_search_strategies', {}).get('default')
        available_rag_search = config_data.get('rag_search_strategies', {}).get('available', [])
        print(f"Default RAG Search: {default_rag_search}")
        for rag_strat in available_rag_search:
            print(f"  - Name: {rag_strat.get('name')}, Description: {rag_strat.get('description')}")


        print("\n--- LLM Config ---")
        llm_config = config_data.get('llm_config', {})
        default_llm_provider = llm_config.get('default_provider')
        llm_providers = llm_config.get('providers', [])
        print(f"Default LLM Provider: {default_llm_provider}")
        for provider in llm_providers:
            print(f"  - Name: {provider.get('name')}, Type: {provider.get('type')}, Model: {provider.get('model')}")


    except StrategyConfigError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
