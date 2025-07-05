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
    except ImportError:
        # If still not found, use a default list and show a warning.
        # This is a fallback for environments where imports are tricky.
        AVAILABLE_RAG_STRATEGIES = ["basic"] # Default to only "basic"
        st.warning("Could not load RAG strategies from core module. Defaulting to 'basic'. Ensure PYTHONPATH is set correctly if more strategies are expected.")


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
                    "rag_strategy": selected_strategy # Get from sidebar
                }
                # TODO: Add data_source_ids selection if needed in the future
                # payload["data_source_ids"] = [...]

                resp = requests.post(f"{BACKEND_URL}/chat/query/", json=payload, headers=headers, timeout=60)

                if resp.status_code == 200:
                    response_data = resp.json()
                    answer = response_data.get("answer", "(No answer received)")
                    strategy_used = response_data.get("strategy_used", selected_strategy) # Use backend confirmed strategy
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "strategy_used": strategy_used
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
        st.write("アップロードしたドキュメントはRAG検索対象になります。TXT/MD/PDF対応。")

        # --- ドキュメントアップロード ---
        with st.form("upload_form", clear_on_submit=True):
            uploaded_files = st.file_uploader("Upload documents", type=["txt", "md", "pdf"], accept_multiple_files=True)
            upload_submit = st.form_submit_button("Upload")
        if upload_submit and uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    resp = requests.post(f"{BACKEND_URL}/documents/upload/", headers=headers, files=files, timeout=120)
                    if resp.status_code == 200:
                        st.success(f"Upload succeeded: {uploaded_file.name}")
                    else:
                        st.error(f"Upload failed: {resp.text}")
                except Exception as e:
                    st.error(f"Upload error: {e}")
            st.session_state.chat_loading = False  # ← 追加: アップロード後にchat_loadingを必ずFalseに
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
