import httpx
from typing import List, Dict, Any, Optional

class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.Client(timeout=timeout)
        self.token: Optional[str] = None

    def _get_auth_headers(self) -> Dict[str, str]:
        if not self.token:
            raise Exception("Not authenticated")
        return {"Authorization": f"Bearer {self.token}"}

    def signup(self, email: str, password: str) -> bool:
        try:
            response = self.client.post(
                f"{self.base_url}/users/",
                json={"email": email, "password": password}
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            print(f"Signup failed: {e.response.text}")
            return False

    def login(self, email: str, password: str) -> bool:
        try:
            response = self.client.post(
                f"{self.base_url}/token",
                data={"username": email, "password": password}
            )
            response.raise_for_status()
            self.token = response.json()["access_token"]
            return True
        except httpx.HTTPStatusError:
            return False

    def get_datasets(self) -> List[Dict[str, Any]]:
        headers = self._get_auth_headers()
        response = self.client.get(f"{self.base_url}/users/me/datasets/", headers=headers)
        response.raise_for_status()
        return response.json()

    def create_dataset(self, name: str, description: str) -> Dict[str, Any]:
        headers = self._get_auth_headers()
        response = self.client.post(
            f"{self.base_url}/users/me/datasets/",
            json={"name": name, "description": description},
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def delete_dataset(self, dataset_id: int) -> bool:
        headers = self._get_auth_headers()
        response = self.client.delete(
            f"{self.base_url}/users/me/datasets/{dataset_id}",
            headers=headers
        )
        response.raise_for_status()
        return True

    def get_documents(self, dataset_id: int) -> List[Dict[str, Any]]:
        headers = self._get_auth_headers()
        response = self.client.get(
            f"{self.base_url}/users/me/datasets/{dataset_id}/documents/",
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def upload_documents(self, dataset_id: int, files: List[Any], strategy: str = "basic") -> List[Dict[str, Any]]:
        headers = self._get_auth_headers()
        file_list = [("files", (file.name, file, file.type)) for file in files]
        
        if strategy == "parent_document":
            endpoint = f"{self.base_url}/users/me/datasets/{dataset_id}/documents/upload_for_parent_document/"
        else:
            endpoint = f"{self.base_url}/users/me/datasets/{dataset_id}/documents/upload_batch/"

        response = self.client.post(
            endpoint,
            files=file_list,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def delete_document(self, dataset_id: int, document_id: int) -> bool:
        headers = self._get_auth_headers()
        response = self.client.delete(
            f"{self.base_url}/users/me/datasets/{dataset_id}/documents/{document_id}",
            headers=headers
        )
        response.raise_for_status()
        return True

    def get_rag_strategies(self) -> List[str]:
        headers = self._get_auth_headers()
        response = self.client.get(f"{self.base_url}/chat/strategies/", headers=headers)
        response.raise_for_status()
        return response.json().get("strategies", [])

    def query_rag(self, query: str, dataset_ids: List[int], strategy: str) -> Dict[str, Any]:
        headers = self._get_auth_headers()
        payload = {
            "query": query,
            "dataset_ids": dataset_ids,
            "strategy": strategy,
        }
        response = self.client.post(
            f"{self.base_url}/chat/query/",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()