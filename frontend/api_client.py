import httpx
from typing import List, Dict, Any, Optional

class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client()
        self.token: Optional[str] = None

    def _get_auth_headers(self) -> Dict[str, str]:
        if not self.token:
            raise Exception("Not authenticated")
        return {"Authorization": f"Bearer {self.token}"}

    def login(self, email: str, password: str) -> bool:
        response = self.client.post(
            f"{self.base_url}/token",
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            return True
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

    def upload_document(self, dataset_id: int, file) -> Dict[str, Any]:
        headers = self._get_auth_headers()
        files = {"file": (file.name, file, file.type)}
        response = self.client.post(
            f"{self.base_url}/users/me/datasets/{dataset_id}/documents/upload/",
            files=files,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    def query_rag(self, query: str, dataset_ids: List[int]) -> Dict[str, Any]:
        headers = self._get_auth_headers()
        response = self.client.post(
            f"{self.base_url}/chat/query/",
            json={"query": query, "dataset_ids": dataset_ids},
            headers=headers
        )
        response.raise_for_status()
        return response.json()
