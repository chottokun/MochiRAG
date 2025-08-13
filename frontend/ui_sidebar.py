import streamlit as st

def render_sidebar():
    st.sidebar.title("MochiRAG Control Panel")

    # --- Dataset Management ---
    st.sidebar.header("Datasets")

    # Fetch and display datasets
    try:
        st.session_state.datasets = st.session_state.api_client.get_datasets()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch datasets: {e}")
        st.session_state.datasets = []

    dataset_names = [d["name"] for d in st.session_state.datasets]
    selected_dataset_name = st.sidebar.selectbox("Select a dataset", dataset_names)

    if selected_dataset_name:
        selected_dataset = next((d for d in st.session_state.datasets if d["name"] == selected_dataset_name), None)
        if selected_dataset:
            st.session_state.selected_dataset_id = selected_dataset["id"]

    with st.sidebar.expander("Create New Dataset"):
        with st.form("new_dataset_form"):
            new_dataset_name = st.text_input("Dataset Name")
            new_dataset_desc = st.text_area("Description")
            submitted = st.form_submit_button("Create")
            if submitted and new_dataset_name:
                with st.spinner("Creating dataset..."):
                    st.session_state.api_client.create_dataset(new_dataset_name, new_dataset_desc)
                    st.rerun()

    # --- Document Upload ---
    if "selected_dataset_id" in st.session_state and st.session_state.selected_dataset_id:
        st.sidebar.header("Upload Document")
        uploaded_file = st.sidebar.file_uploader(
            f"Upload to '{selected_dataset_name}'", 
            type=["txt", "md", "pdf"]
        )
        if uploaded_file is not None:
            with st.spinner("Uploading and processing file..."):
                try:
                    st.session_state.api_client.upload_document(
                        st.session_state.selected_dataset_id, 
                        uploaded_file
                    )
                    st.sidebar.success("File uploaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Upload failed: {e}")

    # --- Logout ---
    st.sidebar.header("") # Spacer
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
