import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# The FastAPI app instance
from backend.main import app

# We need to mock the dependencies that perform external calls
from backend.main import get_current_user
from core.rag_chain_service import rag_chain_service
from core.context_evolution_service import context_evolution_service

# Mock user data
mock_user = MagicMock()
mock_user.id = 1
mock_user.email = "test@example.com"

# Override the dependency to return our mock user
def override_get_current_user():
    return mock_user

app.dependency_overrides[get_current_user] = override_get_current_user

class TestChatEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch.object(rag_chain_service, 'get_rag_response')
    @patch.object(context_evolution_service, 'evolve_context_from_interaction')
    def test_query_with_ace_strategy_triggers_evolution(self, mock_evolve, mock_rag_response):
        # --- Setup ---
        # 1. Configure the mock for the RAG service response
        mock_response_data = {
            "answer": "The ACE strategy is a self-evolving mechanism.",
            "source_documents": []
        }
        mock_rag_response.return_value = MagicMock(**mock_response_data)

        # 2. Define the request payload
        request_payload = {
            "query": "What is the ACE strategy?",
            "dataset_ids": [1],
            "strategy": "ace"
        }

        # --- Execution ---
        # 3. Make a POST request to the chat query endpoint
        response = self.client.post("/chat/query/", json=request_payload)

        # --- Assertions ---
        # 4. Check that the API call was successful
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["answer"], mock_response_data["answer"])

        # 5. Verify that the RAG service was called correctly
        mock_rag_response.assert_called_once()
        # You can add more detailed argument checks here if needed

        # 6. Crucially, verify that the context evolution service was triggered in the background
        mock_evolve.assert_called_once_with(
            user_id=mock_user.id,
            question=request_payload["query"],
            answer=mock_response_data["answer"]
        )

if __name__ == '__main__':
    unittest.main()
