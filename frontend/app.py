import streamlit as st
import requests
import json # For handling potential JSON decode errors more gracefully

import streamlit as st
import requests
import json # For handling potential JSON decode errors more gracefully

# Attempt to import AVAILABLE_RAG_STRATEGIES for the dropdown
# This requires the core module to be accessible.
# If running Streamlit from project root, and core is in PYTHONPATH, this should work.
try:
    from core.rag_chain import AVAILABLE_RAG_STRATEGIES
except ImportError:
    # Fallback if core.rag_chain is not directly importable
    # This might happen if Streamlit is run from the frontend directory directly
    # or if PYTHONPATH is not set up.
    # Add the project root to sys.path if 'core' is not found.
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from core.rag_chain import AVAILABLE_RAG_STRATEGIES
        from core.embedding_manager import embedding_manager # EmbeddingManagerのインスタンス
        from core.chunking_manager import chunking_manager # ChunkingManagerのインスタンス
    except ImportError:
        AVAILABLE_RAG_STRATEGIES = ["basic"]
        st.warning("Could not load RAG, Embedding, or Chunking strategies from core modules. Defaulting to 'basic'. Ensure PYTHONPATH is set correctly.")
        # マネージャーがロードできない場合のフォールバック
        embedding_manager = None
        chunking_manager = None


# --- Configuration ---
BACKEND_URL = "http://localhost:8000" # Ensure this matches your FastAPI backend URL

# --- Session State Initialization ---
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state: # To store user details like username, email
    st.session_state.user = None
if "page" not in st.session_state: # Handles current view: login, register, or app pages
    st.session_state.page = "login"
if "main_app_page" not in st.session_state: # Specific page within the main app
    st.session_state.main_app_page = "Chat"
    # For Document Management Page
    if "datasets" not in st.session_state:
        st.session_state.datasets = [] # List of dataset dicts
    if "selected_dataset_id" not in st.session_state:
        st.session_state.selected_dataset_id = None
    if "files_in_selected_dataset" not in st.session_state:
        st.session_state.files_in_selected_dataset = []


# --- Authentication Functions ---
def login(username, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state.token = token

            # Fetch user details
            headers = {"Authorization": f"Bearer {token}"}
            user_response = requests.get(f"{BACKEND_URL}/users/me", headers=headers)
            if user_response.status_code == 200:
                st.session_state.user = user_response.json()
                st.session_state.page = "app_main" # Navigate to main app view
                st.rerun() # Use st.rerun() which is preferred
            else:
                st.error(f"Failed to fetch user details: {user_response.text}")
                st.session_state.token = None # Clear token if user details fetch fails
                return False
            return True
        else:
            st.error(f"Login failed: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend. Please ensure it's running.")
        return False
    except json.JSONDecodeError:
        st.error("Received an invalid response from the server.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during login: {e}")
        return False

def register(username, email, password):
    try:
        user_data = {"username": username, "email": email, "password": password}
        response = requests.post(f"{BACKEND_URL}/users/", json=user_data)

        if response.status_code == 200: # FastAPI usually returns 200 for POST on success
            st.success("Registration successful! Please login.")
            st.session_state.page = "login" # Redirect to login page
            st.rerun()
            return True
        else:
            try:
                detail = response.json().get("detail", response.text)
            except json.JSONDecodeError:
                detail = response.text
            st.error(f"Registration failed: {detail}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend. Please ensure it's running.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during registration: {e}")
        return False

def logout():
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.page = "login"
    # st.success("Logged out successfully!") # Optional feedback
    st.rerun()

# --- UI Rendering ---

# 1. Login/Register View
if st.session_state.token is None:
    st.sidebar.title("MochiRAG")
    auth_page_choice = st.sidebar.radio("Choose Action", ["Login", "Register"],
                                        index=0 if st.session_state.page == "login" else 1)

    if auth_page_choice == "Login":
        st.session_state.page = "login" # Ensure page state is consistent
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if not username or not password:
                    st.warning("Please enter both username and password.")
                else:
                    login(username, password)

    elif auth_page_choice == "Register":
        st.session_state.page = "register" # Ensure page state is consistent
        st.header("Register")
        with st.form("register_form"):
            reg_username = st.text_input("Username")
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
            reg_submitted = st.form_submit_button("Register")
            if reg_submitted:
                if not reg_username or not reg_email or not reg_password:
                    st.warning("Please fill all fields.")
                else:
                    register(reg_username, reg_email, reg_password)

# 2. Main Application View (after login)
else:
    st.sidebar.title("MochiRAG Menu")
    if st.session_state.user:
        st.sidebar.write(f"Welcome, {st.session_state.user.get('username', 'User')}!")

    st.session_state.main_app_page = st.sidebar.radio(
        "Navigate",
        ["Chat", "Document Management"],
        key="main_nav" # Add a key to avoid issues with radio button state
    )

    if st.sidebar.button("Logout"):
        logout()

    # Main content area based on navigation
    if st.session_state.main_app_page == "Chat":
        st.title("Chat Page")

        # --- RAG Strategy Selection ---
        st.sidebar.subheader("RAG Strategy")
        # Use AVAILABLE_RAG_STRATEGIES imported or defaulted earlier
        selected_strategy = st.sidebar.selectbox(
            "Choose a RAG strategy:",
            options=AVAILABLE_RAG_STRATEGIES,
            index=AVAILABLE_RAG_STRATEGIES.index("basic") if "basic" in AVAILABLE_RAG_STRATEGIES else 0, # Default to basic
            key="rag_strategy_selector"
        )
        st.sidebar.caption(f"Current strategy: **{selected_strategy}**")

        # Option to show/hide references
        if "show_references" not in st.session_state:
            st.session_state.show_references = False # Default to not showing references
        st.session_state.show_references = st.sidebar.checkbox(
            "Show references/sources",
            value=st.session_state.show_references,
            key="show_references_checkbox"
        )

        # --- Dataset Selection for Chat ---
        st.sidebar.subheader("Target Datasets for Chat")
        if not st.session_state.datasets: # Ensure datasets are loaded
            get_user_datasets() # Assuming get_user_datasets is defined globally or accessible

        dataset_options = {ds['name']: ds['dataset_id'] for ds in st.session_state.datasets}
        selected_dataset_names = st.sidebar.multiselect(
            "Select datasets to query (default: all):",
            options=list(dataset_options.keys()),
            key="chat_dataset_selector"
        )
        selected_dataset_ids_for_query = [dataset_options[name] for name in selected_dataset_names if name in dataset_options]


        # --- Chat Session State ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []  # List of dicts: {"role": "user"|"assistant", "content": str, "strategy_used": Optional[str]}
        if "chat_loading" not in st.session_state:
            st.session_state.chat_loading = False

        # --- Display Chat History ---
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                ai_response_text = f"<span style='color: #2b7cff'><b>AI</b>"
                if msg.get("strategy_used"):
                    ai_response_text += f" (<i>{msg['strategy_used']}</i>)"
                ai_response_text += f":</span> {msg['content']}"
                st.markdown(ai_response_text, unsafe_allow_html=True)

                # Display sources if available and checkbox is checked
                if st.session_state.show_references and msg.get("sources"):
                    with st.expander("View Sources", expanded=False): # Collapsible section for sources
                        for i, src in enumerate(msg["sources"]):
                            source_display_text = f"**Source {i+1}:**"
                            metadata = src.get("metadata", {})
                            original_filename = metadata.get("original_filename")
                            data_source_id = metadata.get("data_source_id")
                            page_num = metadata.get("page")

                            if original_filename:
                                source_display_text += f" `{original_filename}`"
                            elif data_source_id: # Fallback to data_source_id if no filename
                                source_display_text += f" ID: `{data_source_id}`"

                            if page_num is not None:
                                source_display_text += f" (Page: {page_num + 1})" # Assuming page is 0-indexed

                            st.markdown(source_display_text, unsafe_allow_html=True)

                            # Optionally display a snippet of page_content
                            page_content_snippet = src.get("page_content", "")[:150] # Show first 150 chars
                            if page_content_snippet:
                                st.caption(f"> {page_content_snippet}...")
                        st.markdown("---")


        # --- Chat Input ---
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Your question", key="chat_input", height=80)
            # selected_strategy is already available from the sidebar selectbox
            submitted = st.form_submit_button("Send", disabled=st.session_state.chat_loading)

        if submitted and user_input.strip():
            st.session_state.chat_loading = True
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                # Include the selected RAG strategy in the payload
                payload = {
                    "question": user_input,
                    "rag_strategy": selected_strategy, # Get from sidebar
                    # Add selected dataset_ids to the payload if any are selected
                }
                if selected_dataset_ids_for_query:
                    payload["dataset_ids"] = selected_dataset_ids_for_query
                # If specific data_source_ids are ever needed, they could be added here too,
                # but backend logic prioritizes data_source_ids over dataset_ids if both are present.
                # For this UI, we primarily use dataset_ids.

                resp = requests.post(f"{BACKEND_URL}/chat/query/", json=payload, headers=headers, timeout=60)

                if resp.status_code == 200:
                    response_data = resp.json()
                    answer = response_data.get("answer", "(No answer received)")
                    strategy_used = response_data.get("strategy_used", selected_strategy) # Use backend confirmed strategy
                    sources_data = response_data.get("sources", None) # Get sources from response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "strategy_used": strategy_used,
                        "sources": sources_data # Store sources in chat history
                    })
                else:
                    error_detail = resp.text
                    try: # Try to parse JSON for more specific error
                        error_detail = resp.json().get("detail", resp.text)
                    except json.JSONDecodeError:
                        pass
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"(Error: {error_detail})",
                        "strategy_used": selected_strategy # Show strategy attempted
                    })
            except requests.exceptions.RequestException as e: # More specific for network/request errors
                 st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"(Request Error: {e})",
                    "strategy_used": selected_strategy
                })
            except Exception as e: # General fallback
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"(Unexpected Error: {e})",
                    "strategy_used": selected_strategy
                })
            finally:
                st.session_state.chat_loading = False
                st.rerun()

        if st.session_state.chat_loading:
            st.info(f"AI is thinking (using {selected_strategy} strategy)...")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    elif st.session_state.main_app_page == "Document Management":
        st.title("Document Management")

        # --- Helper functions for API calls related to datasets and files ---
        def get_user_datasets():
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                response = requests.get(f"{BACKEND_URL}/users/me/datasets/", headers=headers)
                if response.status_code == 200:
                    st.session_state.datasets = response.json()
                    return True
                else:
                    st.error(f"Failed to fetch datasets: {response.status_code} - {response.text}")
                    st.session_state.datasets = []
                    return False
            except Exception as e:
                st.error(f"Error fetching datasets: {e}")
                st.session_state.datasets = []
                return False

        def create_new_dataset(name: str, description: Optional[str] = None):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            payload = {"name": name, "description": description or ""}
            try:
                response = requests.post(f"{BACKEND_URL}/users/me/datasets/", headers=headers, json=payload)
                if response.status_code == 201: # Created
                    st.success(f"Dataset '{name}' created successfully!")
                    get_user_datasets() # Refresh dataset list
                    return True
                else:
                    st.error(f"Failed to create dataset: {response.status_code} - {response.text}")
                    return False
            except Exception as e:
                st.error(f"Error creating dataset: {e}")
                return False

        def delete_selected_dataset(dataset_id_to_delete: str):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                response = requests.delete(f"{BACKEND_URL}/users/me/datasets/{dataset_id_to_delete}/", headers=headers)
                if response.status_code == 204: # No Content
                    st.success(f"Dataset deleted successfully!")
                    st.session_state.selected_dataset_id = None # Clear selection
                    st.session_state.files_in_selected_dataset = []
                    get_user_datasets() # Refresh dataset list
                    return True
                else:
                    st.error(f"Failed to delete dataset: {response.status_code} - {response.text}")
                    return False
            except Exception as e:
                st.error(f"Error deleting dataset: {e}")
                return False

        # --- Dataset Management Section ---
        st.subheader("Your Datasets")

        # Create new dataset form
        with st.expander("Create New Dataset", expanded=False):
            with st.form("create_dataset_form", clear_on_submit=True):
                new_dataset_name = st.text_input("Dataset Name*", key="new_ds_name")
                new_dataset_desc = st.text_area("Description (optional)", key="new_ds_desc")
                submitted_create_dataset = st.form_submit_button("Create Dataset")
                if submitted_create_dataset:
                    if not new_dataset_name.strip():
                        st.warning("Dataset Name is required.")
                    else:
                        create_new_dataset(new_dataset_name, new_dataset_desc)
                        st.rerun()

        # List existing datasets
        if not st.session_state.datasets: # Initial load or if empty
            get_user_datasets()

        if st.session_state.datasets:
            st.write("Available Datasets:")
            # Use columns for better layout: Name | Description | Actions
            cols_title = st.columns((2, 3, 2, 2)) # Adjust ratios as needed
            cols_title[0].write("**Name**")
            cols_title[1].write("**Description**")
            cols_title[2].write("**Manage Files**")
            cols_title[3].write("**Delete**")

            for ds in st.session_state.datasets:
                cols = st.columns((2, 3, 2, 2))
                cols[0].write(ds.get("name", "N/A"))
                cols[1].write(ds.get("description") or "_No description_")

                if cols[2].button("Manage Files", key=f"manage_ds_{ds['dataset_id']}"):
                    st.session_state.selected_dataset_id = ds["dataset_id"]
                    st.session_state.files_in_selected_dataset = [] # Clear previous files
                    # TODO: Fetch files for this dataset (next step)
                    st.rerun()

                if cols[3].button("Delete Dataset", key=f"delete_ds_{ds['dataset_id']}", type="secondary"):
                    # Simple confirmation for now, ideally a modal or more prominent warning
                    st.warning(f"Are you sure you want to delete dataset '{ds['name']}'? This action cannot be undone and will delete all associated files.")
                    if st.button("Confirm Delete", key=f"confirm_delete_ds_{ds['dataset_id']}", type="primary"):
                        delete_selected_dataset(ds["dataset_id"])
                        st.rerun()
                st.markdown("---")

        else:
            st.info("You don't have any datasets yet. Create one above!")

        st.markdown("---") # Separator

        # --- File Management Section (shown if a dataset is selected) ---
        if st.session_state.selected_dataset_id:
            selected_ds_info = next((ds for ds in st.session_state.datasets if ds['dataset_id'] == st.session_state.selected_dataset_id), None)
            if not selected_ds_info: # Should not happen if selected_dataset_id is valid
                st.error("Selected dataset not found. Please select another dataset.")
                st.session_state.selected_dataset_id = None # Reset
                st.rerun()

            selected_ds_name = selected_ds_info.get("name", "Selected Dataset")
            st.subheader(f"Manage Files in Dataset: \"{selected_ds_name}\"")

            def get_files_for_selected_dataset():
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                try:
                    response = requests.get(f"{BACKEND_URL}/users/me/datasets/{st.session_state.selected_dataset_id}/documents/", headers=headers)
                    if response.status_code == 200:
                        st.session_state.files_in_selected_dataset = response.json()
                    else:
                        st.error(f"Failed to fetch files for dataset: {response.status_code} - {response.text}")
                        st.session_state.files_in_selected_dataset = []
                except Exception as e:
                    st.error(f"Error fetching files for dataset: {e}")
                    st.session_state.files_in_selected_dataset = []

            def delete_file_from_selected_dataset(data_source_id_to_delete: str):
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                try:
                    response = requests.delete(f"{BACKEND_URL}/users/me/datasets/{st.session_state.selected_dataset_id}/documents/{data_source_id_to_delete}/", headers=headers)
                    if response.status_code == 204:
                        st.success(f"File deleted successfully!")
                        get_files_for_selected_dataset() # Refresh file list
                        return True
                    else:
                        st.error(f"Failed to delete file: {response.status_code} - {response.text}")
                        return False
                except Exception as e:
                    st.error(f"Error deleting file: {e}")
                    return False

            # Load files for the selected dataset if not already loaded or if selection changed
            # This condition might need refinement to avoid excessive reloads.
            # For now, simple check: if files list is empty for a selected dataset, try to load.
            if not st.session_state.files_in_selected_dataset and st.session_state.selected_dataset_id:
                 get_files_for_selected_dataset()


            # --- Vector Store Strategy Selection for Upload ---
            st.markdown("#### Upload New Document to this Dataset")
            # These strategy selectors are now part of the file management for a selected dataset
            available_embedding_strategies = ["default"]
            if embedding_manager:
                try:
                    available_embedding_strategies = embedding_manager.get_available_strategies()
                except Exception as e:
                    st.error(f"Failed to load embedding strategies: {e}")

            upload_embedding_strategy = st.selectbox(
                "Choose Embedding Strategy for Upload:",
                options=available_embedding_strategies,
                index=0,
                key="upload_embedding_strategy_selector"
            )

            available_chunking_strategies = ["default"]
            if chunking_manager:
                try:
                    available_chunking_strategies = chunking_manager.get_available_strategies()
                except Exception as e:
                    st.error(f"Failed to load chunking strategies: {e}")

            upload_chunking_strategy = st.selectbox(
                "Choose Chunking Strategy for Upload:",
                options=available_chunking_strategies,
                index=0,
                key="upload_chunking_strategy_selector"
            )
            upload_chunk_size = st.number_input("Chunk Size (if applicable)", value=1000, min_value=100, step=50, key="upload_chunk_size_input")
            upload_chunk_overlap = st.number_input("Chunk Overlap (if applicable)", value=200, min_value=0, step=20, key="upload_chunk_overlap_input")


            # --- File Upload Form ---
            with st.form("upload_file_to_dataset_form", clear_on_submit=True):
                uploaded_file_ds = st.file_uploader("Upload a document (TXT, MD, PDF)", type=["txt", "md", "pdf"], key="file_uploader_ds")
                upload_file_submit_ds = st.form_submit_button("Upload to Dataset")

            if upload_file_submit_ds and uploaded_file_ds:
                if not st.session_state.selected_dataset_id:
                    st.error("No dataset selected for upload.")
                else:
                    upload_params = {
                        "embedding_strategy": upload_embedding_strategy if upload_embedding_strategy != "default" else None,
                        "chunking_strategy": upload_chunking_strategy if upload_chunking_strategy != "default" else None,
                    }
                    # Assuming 'recursive' in strategy name implies these params are relevant
                    if upload_chunking_strategy and "recursive" in upload_chunking_strategy.lower():
                        upload_params["chunking_params"] = {
                            "chunk_size": upload_chunk_size,
                            "chunk_overlap": upload_chunk_overlap
                        }

                    form_data_payload = {
                        'embedding_strategy': upload_params["embedding_strategy"],
                        'chunking_strategy': upload_params["chunking_strategy"],
                    }
                    if upload_params.get("chunking_params"):
                        form_data_payload['chunking_params_json'] = json.dumps(upload_params["chunking_params"])

                    files_payload = {"file": (uploaded_file_ds.name, uploaded_file_ds, uploaded_file_ds.type)}
                    headers_auth = {"Authorization": f"Bearer {st.session_state.token}"}

                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/users/me/datasets/{st.session_state.selected_dataset_id}/documents/upload/",
                            headers=headers_auth,
                            files=files_payload,
                            data=form_data_payload,
                            timeout=120
                        )
                        if resp.status_code == 200: # Expecting 200 from backend upon successful upload and processing
                            st.success(f"Document '{uploaded_file_ds.name}' uploaded successfully to dataset '{selected_ds_name}'.")
                            get_files_for_selected_dataset() # Refresh file list
                        else:
                            st.error(f"Upload failed: {resp.status_code} - {resp.text}")
                    except Exception as e:
                        st.error(f"Error during upload: {e}")
                    st.rerun()

            st.markdown("---")
            st.markdown("#### Documents in this Dataset")
            if st.button("Refresh File List", key="refresh_files_btn"):
                get_files_for_selected_dataset()
                st.rerun()

            if st.session_state.files_in_selected_dataset:
                # Display files in columns
                cols_files_title = st.columns((3, 2, 2, 1))
                cols_files_title[0].write("**Filename**")
                cols_files_title[1].write("**Uploaded At**")
                cols_files_title[2].write("**Chunks**")
                cols_files_title[3].write("**Delete**")

                for f_meta in st.session_state.files_in_selected_dataset:
                    cols_f = st.columns((3, 2, 2, 1))
                    cols_f[0].write(f_meta.get("original_filename", "N/A"))
                    cols_f[1].write(f_meta.get("uploaded_at", "-").split("T")[0]) # Just date for brevity
                    cols_f[2].write(str(f_meta.get("chunk_count", "-")))
                    if cols_f[3].button("🗑️", key=f"delete_file_{f_meta['data_source_id']}", help="Delete this file"):
                        # Simple confirmation, could be improved with a modal
                        st.warning(f"Delete file '{f_meta['original_filename']}'?")
                        if st.button("Confirm Delete File", key=f"confirm_delete_file_{f_meta['data_source_id']}", type="primary"):
                           delete_file_from_selected_dataset(f_meta["data_source_id"])
                           st.rerun()
                st.markdown("---")
            else:
                st.info("No documents found in this dataset. Upload one above.")

        else: # No dataset selected
            st.info("Select a dataset from the list above to manage its files or upload new documents.")


        # --- Vector Store Strategy Selection (This part will be for file upload within a selected dataset) ---
        # This section needs to be moved/adapted to be part of the "File Management Section"
        # st.subheader("Vector Store Processing Strategies") # Keep this for context if file upload is shown

        # available_embedding_strategies = ["default"]
        # if embedding_manager:
        #     try:
        #         available_embedding_strategies = embedding_manager.get_available_strategies()
            except Exception as e:
                st.error(f"Failed to load embedding strategies: {e}")

        selected_embedding_strategy = st.selectbox(
            "Choose Embedding Strategy:",
            options=available_embedding_strategies,
            index=0,
            key="embedding_strategy_selector"
        )

        available_chunking_strategies = ["default"]
        if chunking_manager:
            try:
                available_chunking_strategies = chunking_manager.get_available_strategies()
            except Exception as e:
                st.error(f"Failed to load chunking strategies: {e}")

        selected_chunking_strategy = st.selectbox(
            "Choose Chunking Strategy:",
            options=available_chunking_strategies,
            index=0,
            key="chunking_strategy_selector"
        )

        # チャンキングパラメータ入力 (例: chunk_size)
        # chunk_params_json = st.text_input("Chunking Parameters (JSON string, optional)", value='{"chunk_size": 1000, "chunk_overlap": 200}')
        # よりユーザーフレンドリーなUIも検討可能 (例: 数値入力フィールド)
        custom_chunk_size = st.number_input("Chunk Size (for recursive)", value=1000, min_value=100, step=50, key="chunk_size_input")
        custom_chunk_overlap = st.number_input("Chunk Overlap (for recursive)", value=200, min_value=0, step=20, key="chunk_overlap_input")


        # --- ドキュメントアップロード ---
        st.subheader("Upload New Document")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf"], accept_multiple_files=False) # 一旦単一ファイルに
            upload_submit = st.form_submit_button("Upload Document")

        if upload_submit and uploaded_file:
            # 選択された戦略とパラメータを準備
            upload_request_data = {
                "embedding_strategy": selected_embedding_strategy if selected_embedding_strategy != "default" else None,
                "chunking_strategy": selected_chunking_strategy if selected_chunking_strategy != "default" else None,
            }
            # RecursiveTextSplitterの場合のパラメータ設定（他の戦略では異なるUIが必要になる可能性）
            if selected_chunking_strategy and "recursive" in selected_chunking_strategy:
                 upload_request_data["chunking_params"] = {
                     "chunk_size": custom_chunk_size,
                     "chunk_overlap": custom_chunk_overlap
                 }

            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                # ファイルとリクエストデータを一緒に送信 (multipart/form-data)
                # FastAPI側で Body() と File() を同時に受け取るには工夫が必要な場合がある
                # ここでは、upload_request_data をJSON文字列としてファイルと一緒に送信する
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type),
                    # DocumentUploadRequestをJSON文字列として送信
                    "upload_request": (None, json.dumps(upload_request_data), "application/json")
                }

                # バックエンドAPIの /documents/upload/ は DocumentUploadRequest を Body(...) で受け取る想定
                # ファイルアップロードとJSONボディを同時に扱うには、FastAPI側で工夫が必要。
                # 一般的なのは、JSONボディをFormデータとして送信するか、
                # リクエストを2段階に分ける（メタ情報POST -> ファイルPUTなど）。
                # ここでは、Body()とFile()を同時に受け付けるようにFastAPIが設定されていると仮定する。
                # しかし、requestsライブラリでこれを単純に行うのは難しい。
                # Bodyパラメータは通常 application/json として送られる。
                # 解決策: FastAPIエンドポイントを修正し、戦略パラメータをフォームデータとして受け取る。
                # または、戦略パラメータをクエリパラメータとして渡す。今回はフォームデータとして修正する想定で進める。

                form_data = {
                    'embedding_strategy': upload_request_data["embedding_strategy"],
                    'chunking_strategy': upload_request_data["chunking_strategy"],
                }
                if upload_request_data.get("chunking_params"):
                    form_data['chunking_params_json'] = json.dumps(upload_request_data["chunking_params"])


                # `files` に `upload_request` を含めず、`data` でフォームデータを渡す
                actual_files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                resp = requests.post(
                    f"{BACKEND_URL}/documents/upload/",
                    headers=headers,
                    files=actual_files, # ファイルのみ
                    data=form_data,     # 戦略パラメータをフォームデータとして送信
                    timeout=120
                )

                if resp.status_code == 200:
                    st.success(f"Upload succeeded: {uploaded_file.name}")
                else:
                    st.error(f"Upload failed: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Upload error: {e}")
            # st.session_state.chat_loading = False # Document Managementページなのでchat_loadingは不要
            st.rerun()

        # --- ドキュメント一覧表示 ---
        st.subheader("Your Uploaded Documents")
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            resp = requests.get(f"{BACKEND_URL}/documents/", headers=headers)
            if resp.status_code == 200:
                docs = resp.json()
                if docs:
                    for doc in docs:
                        st.markdown(f"- **{doc['original_filename']}** (ID: `{doc['data_source_id']}`) - {doc['status']} - {doc['uploaded_at']} - チャンク数: {doc.get('chunk_count', '-')}")
                else:
                    st.info("No documents uploaded yet.")
            else:
                st.error(f"Failed to fetch documents: {resp.text}")
        except Exception as e:
            st.error(f"Error fetching documents: {e}")

    else:
        st.title("Welcome")
        st.write("Select a page from the sidebar.")

# To run this app:
# 1. Ensure the FastAPI backend is running (e.g., uvicorn backend.main:app --reload --port 8000)
# 2. Install frontend requirements: pip install -r frontend/requirements.txt
# 3. Run Streamlit: streamlit run frontend/app.py
