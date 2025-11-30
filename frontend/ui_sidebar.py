import streamlit as st
import json
import os

def get_shared_dbs():
    """Reads the shared DBs from the json file."""
    # Try a simple relative path first, as Streamlit's CWD is usually the project root
    shared_dbs_path = 'shared_dbs.json'
    
    if not os.path.exists(shared_dbs_path):
        return []
    
    try:
        with open(shared_dbs_path, 'r') as f:
            shared_dbs_data = json.load(f)
            # Handle both list of strings and list of objects
            if shared_dbs_data and isinstance(shared_dbs_data[0], dict):
                names = [db['name'] for db in shared_dbs_data]
            else:
                names = shared_dbs_data # Assumes it's a list of strings
            return names
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        st.sidebar.error(f"Error reading or parsing shared_dbs.json: {e}")
        return []

def handle_file_uploads(dataset_id: int, uploader_key: str, strategy: str):
    """Callback function to handle multiple file uploads with a specific strategy."""
    uploaded_files = st.session_state.get(uploader_key)
    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} file(s) with '{strategy}' strategy..."):
            try:
                st.session_state.api_client.upload_documents(dataset_id, uploaded_files, strategy)
                st.success("File(s) uploaded successfully!")
            except Exception as e:
                st.error(f"Upload failed: {e}")

def render_data_management():
    st.sidebar.header("Data Management")
    shared_dbs = get_shared_dbs()

    # --- Dataset Creation ---
    with st.sidebar.expander("Create New Dataset"):
        with st.form("new_dataset_form", clear_on_submit=True):
            new_dataset_name = st.text_input("Dataset Name")
            new_dataset_desc = st.text_area("Description")
            submitted = st.form_submit_button("Create")
            if submitted and new_dataset_name:
                try:
                    st.session_state.api_client.create_dataset(new_dataset_name, new_dataset_desc)
                    st.sidebar.success("Dataset created!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Creation failed: {e}")

    # --- Dataset and Document Listing ---
    try:
        st.session_state.datasets = st.session_state.api_client.get_datasets()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch datasets: {e}")
        st.session_state.datasets = []

    

    if not st.session_state.datasets:
        st.sidebar.info("No datasets found. Create one to get started.")
        return

    for dataset in st.session_state.datasets:
        is_shared = dataset['name'] in shared_dbs
        with st.sidebar.expander(dataset['name']):
            st.write(f"_{dataset.get('description', 'No description')}_")
            if is_shared:
                st.info("This is a shared dataset. Management is restricted.")

            # --- Ingestion Strategy Selection ---
            ingestion_strategy = st.selectbox(
                "Select Ingestion Strategy",
                options=["basic", "parent_document"],
                key=f"ingestion_strategy_{dataset['id']}"
            )

            # --- Document Upload (using callback) ---
            if not is_shared:
                uploader_key = f"upload_{dataset['id']}"
                st.file_uploader(
                    f"Upload to '{dataset['name']}'",
                    type=["txt", "md", "pdf"],
                    key=uploader_key,
                    on_change=handle_file_uploads,
                    args=(dataset['id'], uploader_key, ingestion_strategy),
                    accept_multiple_files=True
                )

            # --- Document List ---
            try:
                documents = st.session_state.api_client.get_documents(dataset['id'])
                if documents:
                    st.markdown("---")
                    st.write("**Documents:**")
                    for doc in documents:
                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            st.text(doc.get('original_filename', 'N/A'))
                        if not is_shared:
                            with col2:
                                if st.button("🗑️", key=f"del_doc_{doc['id']}", help="Delete document"):
                                    try:
                                        st.session_state.api_client.delete_document(dataset['id'], doc['id'])
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed: {e}")
                else:
                    st.info("No documents in this dataset.")
            except Exception as e:
                st.error(f"Could not load documents: {e}")

            # --- Dataset Deletion ---
            if not is_shared:
                st.markdown("---")
                if st.button("Delete this Dataset", key=f"del_ds_{dataset['id']}", type="primary"):
                    try:
                        st.session_state.api_client.delete_dataset(dataset['id'])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deletion failed: {e}")

def render_sidebar():
    st.sidebar.title("MochiRAG Control Panel")

    # --- Settings ---
    st.sidebar.header("Settings")
    st.session_state.api_timeout = st.number_input(
        "API Timeout (seconds)", 
        min_value=5, 
        max_value=300, 
        value=st.session_state.get("api_timeout", 30), 
        step=5
    )

    render_data_management()

    # --- Logout ---
    st.sidebar.header("") # Spacer
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
