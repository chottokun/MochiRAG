Code Review
This pull request introduces a significant enhancement to the retriever functionality by implementing an EnsembleRetriever to allow simultaneous searching across multiple personal and shared datasets. The implementation in BasicRetrieverStrategy is well-structured. The PR also includes the removal of the HydeRetrieverStrategy due to compatibility issues, along with new tests, build scripts, and updated documentation. My review focuses on improving code quality, fixing bugs in a utility script and a test case, and enhancing the maintainability of the core logic and build process.

core/retriever_manager.py
Comment on lines +29 to +31
import importlib
import inspect
import logging
Contributor
@gemini-code-assist gemini-code-assist bot 48 minutes ago
high

This file has several issues related to imports and logging that should be addressed to improve maintainability:

There are multiple duplicated import blocks for importlib, inspect, and logging (here, lines 17-19, and 291-293). All imports should be consolidated at the top of the file.
The logger instance is defined on line 295. It should be defined at the top of the file, right after the imports, as per standard Python conventions (logger = logging.getLogger(__name__)).
print() is used for logging warnings on lines 115 and 126. Please use logger.warning() instead for proper, configurable logging.
Cleaning this up will make the code much cleaner and more robust.

@chottokun	Reply...
core/retriever_manager.py
Comment on lines +240 to +267
def _normalize_retriever_obj(obj: BaseRetriever) -> BaseRetriever:
    """Ensure returned retriever(s) are proper Runnable/BaseRetriever instances.
    If an EnsembleRetriever contains non-Runnable retrievers (e.g., MagicMocks),
    wrap them in _BaseRetrieverAdapter so pydantic/type checks pass at runtime.
    """
    # Import locally to avoid circular issues with typing/runtime
    from langchain.retrievers import EnsembleRetriever as _Ensemble

    if isinstance(obj, _Ensemble):
        new_retrievers = []
        for r in obj.retrievers:
            if isinstance(r, Runnable):
                new_retrievers.append(r)
            else:
                new_retrievers.append(_BaseRetrieverAdapter(r))

            # Create a shallow copy of EnsembleRetriever with normalized retrievers
            # Ensure weights is a proper list (pydantic expects list[float])
            weights = getattr(obj, "weights", None)
            if weights is None:
                weights = [1.0 / len(new_retrievers)] * len(new_retrievers)
            return _Ensemble(retrievers=new_retrievers, weights=weights)

    # Single retriever
    if isinstance(obj, Runnable):
        return obj
    return _BaseRetrieverAdapter(obj)
Contributor
@gemini-code-assist gemini-code-assist bot 48 minutes ago
high

This function _normalize_retriever_obj appears to be unused in the current codebase. It also contains a critical bug where the return statement is inside the for loop (line 262), which would cause it to exit prematurely after processing only the first sub-retriever of an ensemble. Since similar normalization logic is already correctly implemented within BasicRetrieverStrategy.get_retriever, consider removing this unused and buggy function to avoid confusion and future maintenance issues.

@chottokun	Reply...
scripts/push-image.sh
Comment on lines +12 to +17
IMAGE=${IMAGE:-mochirag-backend}

FULL_NAME="$REGISTRY/$OWNER/$REPO/$IMAGE:$TAG"

echo "Building $FULL_NAME"
docker build -f Dockerfile.backend -t "$FULL_NAME" .
Contributor
@gemini-code-assist gemini-code-assist bot 48 minutes ago
high

This script is intended to push an image, but it has Dockerfile.backend hardcoded while using an IMAGE variable. This will cause incorrect behavior if you try to push the frontend image by setting IMAGE=mochirag-frontend. The Dockerfile used for the build should be determined based on the IMAGE variable to make the script reusable for both backend and frontend.

Suggested change
IMAGE=${IMAGE:-mochirag-backend}
FULL_NAME="$REGISTRY/$OWNER/$REPO/$IMAGE:$TAG"
echo "Building $FULL_NAME"
docker build -f Dockerfile.backend -t "$FULL_NAME" .
IMAGE=${IMAGE:-mochirag-backend}
# Determine Dockerfile based on image name
DOCKERFILE="Dockerfile.backend"
if [ "$IMAGE" = "mochirag-frontend" ]; then
  DOCKERFILE="Dockerfile.frontend"
fi
FULL_NAME="$REGISTRY/$OWNER/$REPO/$IMAGE:$TAG"
echo "Building $FULL_NAME using $DOCKERFILE"
docker build -f "$DOCKERFILE" -t "$FULL_NAME" .
@chottokun	Reply...
tests/core/test_retriever_manager.py
Comment on lines +51 to +58
    def test_personal_only_single_retriever(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_as_retriever = mock_get_vector_store.return_value.as_retriever
        mock_as_retriever.return_value = self.mock_retriever
        dataset_ids = [1, 2, 3]
        retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertNotIsInstance(retriever, EnsembleRetriever)
        self.assertEqual(retriever, self.mock_retriever)
Contributor
@gemini-code-assist gemini-code-assist bot 48 minutes ago
high

This test case will fail. The get_retriever method wraps non-Runnable retrievers (like the MagicMock used here) in a _BaseRetrieverAdapter. Therefore, the retriever variable will hold an instance of _BaseRetrieverAdapter, not the original self.mock_retriever. The assertion self.assertEqual(retriever, self.mock_retriever) will fail. You should instead assert the type of the returned object and check that its _delegate attribute is the mock retriever.

Suggested change
    def test_personal_only_single_retriever(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_as_retriever = mock_get_vector_store.return_value.as_retriever
        mock_as_retriever.return_value = self.mock_retriever
        dataset_ids = [1, 2, 3]
        retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertNotIsInstance(retriever, EnsembleRetriever)
        self.assertEqual(retriever, self.mock_retriever)
    def test_personal_only_single_retriever(self, mock_get_config, mock_get_vector_store):
        mock_get_config.return_value.parameters.get.return_value = 5
        mock_as_retriever = mock_get_vector_store.return_value.as_retriever
        mock_as_retriever.return_value = self.mock_retriever
        dataset_ids = [1, 2, 3]
        retriever = self.strategy.get_retriever(self.user_id, dataset_ids)
        self.assertNotIsInstance(retriever, EnsembleRetriever)
        from core.retriever_manager import _BaseRetrieverAdapter
        self.assertIsInstance(retriever, _BaseRetrieverAdapter)
        self.assertEqual(retriever._delegate, self.mock_retriever)
@chottokun	Reply...
Makefile
Comment on lines +15 to +21
push-backend:
	docker build -f Dockerfile.backend -t $(BACKEND_IMAGE):$(TAG) .
	docker push $(BACKEND_IMAGE):$(TAG)

push-frontend:
	docker build -f Dockerfile.frontend -t $(FRONTEND_IMAGE):$(TAG) .
	docker push $(FRONTEND_IMAGE):$(TAG)
Contributor
@gemini-code-assist gemini-code-assist bot 48 minutes ago
medium

The push-backend and push-frontend targets currently re-run docker build. This is inefficient as it rebuilds the image from scratch, even if an up-to-date version was just created with make build-backend. It's better practice to separate build and push logic. The push target should tag the existing :local image and push the new tag.

push-backend:
	docker tag $(BACKEND_IMAGE):local $(BACKEND_IMAGE):$(TAG)
	docker push $(BACKEND_IMAGE):$(TAG)

push-frontend:
	docker tag $(FRONTEND_IMAGE):local $(FRONTEND_IMAGE):$(TAG)
	docker push $(FRONTEND_IMAGE):$(TAG)

    