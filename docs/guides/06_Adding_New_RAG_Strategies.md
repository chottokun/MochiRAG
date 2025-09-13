# Guide: Adding New RAG Retriever Strategies

This guide explains how to add a new custom retriever strategy to the RAG system. The system is designed to be extensible, allowing developers to easily add new retrieval methods.

## Overview

The system uses a **Strategy Pattern** to manage different retriever implementations. All retriever strategies are managed by the `RetrieverManager` located in `core/retriever_manager.py`.

The manager dynamically loads retriever strategies based on the configuration in `config/strategies.yaml`. This means that to add a new strategy, you only need to:
1.  Implement the strategy as a new class.
2.  Declare the strategy in the YAML configuration file.

The application will automatically detect and load the new strategy at startup.

## Step 1: Implement the Strategy Class

All retriever strategy classes must inherit from the `RetrieverStrategy` abstract base class and implement the `get_retriever` method. These classes are located in `core/retriever_manager.py`.

### Requirements:
-   **Class Location**: The new class must be added to `core/retriever_manager.py`.
-   **Inheritance**: Must inherit from `RetrieverStrategy`.
-   **Method**: Must implement `get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever`.

### Example:
Here is an example of a simple custom retriever strategy. You can add this class to the `core/retriever_manager.py` file.

```python
# In core/retriever_manager.py

from langchain.retrievers import TFIDFRetriever
from langchain_core.documents import Document

# ... (add with other strategy classes)

class TfidfRetrieverStrategy(RetrieverStrategy):
    """
    A simple strategy that uses TF-IDF to retrieve documents.
    Note: This is a simplified example and assumes documents are available in-memory.
    A real implementation would need to fetch documents from a database.
    """
    def get_retriever(self, user_id: int, dataset_ids: Optional[List[int]] = None) -> BaseRetriever:
        # NOTE: This is a mock implementation for demonstration purposes.
        # A real-world scenario would involve fetching documents for the user/dataset.
        mock_documents = [
            Document(page_content="Mochi is a Japanese rice cake made of mochigome."),
            Document(page_content="Software engineering is a systematic approach to software development."),
            Document(page_content="The MochiRAG system is extensible."),
        ]

        return TFIDFRetriever.from_documents(
            documents=mock_documents,
            k=3
        )

```

## Step 2: Configure the Strategy in YAML

After implementing the class, you need to register it in the `config/strategies.yaml` file under the `retrievers` section.

### Requirements:
-   Add a new key with the desired strategy name (e.g., `tfidf_retriever`).
-   Specify the `strategy_class` key with the exact name of the class you created.
-   Provide a `description` and any necessary `parameters`.

### Example:
Add the following to `config/strategies.yaml`:

```yaml
# In config/strategies.yaml, under the 'retrievers' section

  # ... (other retrievers)

  tfidf_retriever:
    strategy_class: "TfidfRetrieverStrategy"
    description: "A simple TF-IDF based retriever for demonstration."
    parameters: {}

```

## Step 3: Verification

Once you have completed the two steps above, restart the application. The new strategy, "tfidf_retriever", will automatically appear in the "Select RAG Strategy" dropdown in the chat interface.

The `RetrieverManager` handles the loading and instantiation of the new strategy class, making it available for use throughout the application.
