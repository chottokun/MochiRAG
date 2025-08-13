import streamlit as st

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

            # --- Document Upload ---
            uploaded_file = st.file_uploader(
                f"Upload to '{dataset['name']}'",
                type=["txt", "md", "pdf"],
                key=f"upload_{dataset['id']}"
            )
            if uploaded_file:
                with st.spinner("Processing file..."):
                    try:
                        st.session_state.api_client.upload_document(dataset['id'], uploaded_file)
                        st.success("File uploaded!")
                        st.rerun() # Refresh to show new document
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

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

    render_data_management()

    # --- Logout ---
    st.sidebar.header("") # Spacer
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
