import streamlit as st

def handle_file_uploads(dataset_id: int, uploader_key: str):
    """Callback function to handle multiple file uploads."""
    uploaded_files = st.session_state.get(uploader_key)
    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            try:
                st.session_state.api_client.upload_documents(dataset_id, uploaded_files)
                st.success("File(s) uploaded successfully!")
                # To prevent re-triggering, we might need to clear the uploader state
                # st.session_state[uploader_key] = [] # This can be tricky with Streamlit's state
            except Exception as e:
                st.error(f"Upload failed: {e}")

def render_data_management():
    st.sidebar.header("Data Management")

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
        # Note: This could be cached for better performance
        st.session_state.datasets = st.session_state.api_client.get_datasets()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch datasets: {e}")
        st.session_state.datasets = []

    if not st.session_state.datasets:
        st.sidebar.info("No datasets found. Create one to get started.")
        return

    for dataset in st.session_state.datasets:
        with st.sidebar.expander(dataset['name']):
            st.write(f"_{dataset.get('description', 'No description')}_")

            # --- Document Upload (using callback) ---
            uploader_key = f"upload_{dataset['id']}"
            st.file_uploader(
                f"Upload to '{dataset['name']}'",
                type=["txt", "md", "pdf"],
                key=uploader_key,
                on_change=handle_file_uploads,
                args=(dataset['id'], uploader_key),
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