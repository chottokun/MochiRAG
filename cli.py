import click
import json
import os
import re
from core.ingestion_service import ingestion_service, EmbeddingServiceError
from core.vector_store_manager import vector_store_manager

def sanitize_name_for_collection(name: str) -> str:
    """Sanitizes a string to be a valid ChromaDB collection name."""
    # Remove special characters, replace spaces with underscores
    s = re.sub(r'[^\w\s-]', '', name).strip().lower()
    s = re.sub(r'[-\s]+', '_', s)
    # Add the 'shared_' prefix to avoid collisions with user collections
    return f"shared_{s}"

@click.group()
def app():
    """A CLI for managing the MochiRAG application."""
    # This context_obj is a simple way to pass the vector store manager
    # if we add more commands that need it.
    pass

@app.command("create-shared-db")
@click.option("--name", required=True, help="The display name for the shared database.")
@click.option("--source-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="The directory containing the source documents (PDFs, TXTs, etc.).")
def create_shared_db(name, source_dir):
    """Creates a shared vector database from a source directory."""
    click.echo(f"Starting creation of shared database: '{name}'")

    # 1. Initialize vector store client to ensure connection
    try:
        vector_store_manager.initialize_client()
    except Exception as e:
        click.secho(f"Error: Could not connect to vector store. Is it running? Details: {e}", fg="red")
        return

    # 2. Find supported files in the source directory
    supported_extensions = [".pdf", ".txt", ".md"]
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        click.secho(f"Warning: No supported documents found in '{source_dir}'.", fg="yellow")
        return

    click.echo(f"Found {len(file_paths)} documents to ingest.")

    # 3. Ingest the documents into a new collection
    collection_name = sanitize_name_for_collection(name)
    try:
        ingestion_service.ingest_documents_for_shared_db(file_paths, collection_name)
    except EmbeddingServiceError as e:
        click.secho(f"Error during document ingestion: {e}", fg="red")
        return
    except Exception as e:
        click.secho(f"An unexpected error occurred during ingestion: {e}", fg="red")
        return

    # 4. Update the shared_dbs.json file
    shared_dbs_path = "shared_dbs.json"
    try:
        if os.path.exists(shared_dbs_path):
            with open(shared_dbs_path, "r") as f:
                shared_dbs = json.load(f)
        else:
            shared_dbs = []

        # Check for existing name
        if any(db["name"] == name for db in shared_dbs):
            click.secho(f"Error: A shared database with the name '{name}' already exists.", fg="red")
            click.echo("Aborting without changing configuration. The vector data may have been created.")
            return

        # Determine the next available negative ID
        next_id = -1
        if shared_dbs:
            # Find the minimum ID (most negative) and go one lower
            next_id = min(db.get("id", 0) for db in shared_dbs) - 1

        new_db_entry = {
            "id": next_id,
            "name": name,
            "description": f"Shared database created from '{source_dir}'",
            "collection_name": collection_name
        }
        shared_dbs.append(new_db_entry)

        with open(shared_dbs_path, "w") as f:
            json.dump(shared_dbs, f, indent=2)

        click.echo(f"Successfully created and registered shared database: '{name}'")

    except Exception as e:
        click.secho(f"Error updating configuration file '{shared_dbs_path}': {e}", fg="red")
        click.echo("Please note: The vector data may have been created, but the application will not be able to use it.")

if __name__ == '__main__':
    app()
