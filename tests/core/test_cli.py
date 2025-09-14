import pytest
import json
import os
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

# Assume cli.py will be created at the root
from cli import app

@pytest.fixture
def runner():
    return CliRunner()

def test_create_shared_db_help(runner):
    """Test that the CLI command has a help option and exits cleanly."""
    result = runner.invoke(app, ["create-shared-db", "--help"])
    assert result.exit_code == 0
    assert "Usage: app create-shared-db" in result.output
    assert "Creates a shared vector database from a source directory." in result.output

@patch("cli.vector_store_manager")
@patch("cli.ingestion_service")
def test_create_shared_db_success(mock_ingestion_service, mock_vector_store_manager, runner):
    """
    Test the successful creation of a shared database.
    This test is expected to fail until the CLI is implemented.
    """
    # Setup: Create a dummy source directory and a dummy file
    source_dir = "temp_test_docs"
    os.makedirs(source_dir, exist_ok=True)
    with open(os.path.join(source_dir, "test.txt"), "w") as f:
        f.write("This is a test document.")

    # Setup: Create an initial shared_dbs.json
    initial_shared_dbs = [{"id": -1, "name": "Existing DB", "collection_name": "shared_existing_db"}]
    with open("shared_dbs.json", "w") as f:
        json.dump(initial_shared_dbs, f)

    # Execute the CLI command
    db_name = "My New CLI DB"
    result = runner.invoke(app, [
        "create-shared-db",
        "--name", db_name,
        "--source-dir", source_dir,
    ])

    # Assertions
    assert result.exit_code == 0, f"CLI command failed with output: {result.output}"

    # Assert that ingestion_service was called
    mock_ingestion_service.ingest_documents_for_shared_db.assert_called_once()

    # Assert that shared_dbs.json was updated correctly
    with open("shared_dbs.json", "r") as f:
        updated_dbs = json.load(f)

    assert len(updated_dbs) == 2
    new_db_entry = next((item for item in updated_dbs if item["name"] == db_name), None)
    assert new_db_entry is not None
    assert new_db_entry["id"] == -2
    assert new_db_entry["collection_name"] == "shared_my_new_cli_db"

    # Teardown
    os.remove("shared_dbs.json")
    os.remove(os.path.join(source_dir, "test.txt"))
    os.rmdir(source_dir)
