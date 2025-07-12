import streamlit as st
import requests
import json
from typing import List, Dict, Any

# --- Session State Initialization ---
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "login"
if "main_app_page" not in st.session_state:
    st.session_state.main_app_page = "Chat"

if "datasets" not in st.session_state: # Ensure this is always initialized
    st.session_state.datasets = []
if "selected_dataset_id" not in st.session_state:
    st.session_state.selected_dataset_id = None
if "files_in_selected_dataset" not in st.session_state:
    st.session_state.files_in_selected_dataset = []

AVAILABLE_RAG_STRATEGIES = ["basic"]
embedding_manager = None
chunking_manager = None

try:
    from core.rag_chain import AVAILABLE_RAG_STRATEGIES as core_rag_strategies
    from core.embedding_manager import embedding_manager as em
    from core.chunking_manager import chunking_manager as cm
    AVAILABLE_RAG_STRATEGIES = core_rag_strategies
    embedding_manager = em
    chunking_manager = cm
except ImportError:
    # print to console during development if core modules fail to load
    print("Warning: Could not load core RAG, Embedding, or Chunking strategies. Using defaults.")
    pass

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"

# --- Authentication Functions ---
def login(username, password):
    try:
        response = requests.post(f"{BACKEND_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state.token = token
            headers = {"Authorization": f"Bearer {token}"}
            user_response = requests.get(f"{BACKEND_URL}/users/me", headers=headers)
            if user_response.status_code == 200:
                st.session_state.user = user_response.json()
                st.session_state.page = "app_main"
                st.rerun()
            else:
                st.error(f"Failed to fetch user details: {user_response.text}")
                st.session_state.token = None; return False
            return True
        else: st.error(f"Login failed: {response.text}"); return False
    except requests.exceptions.ConnectionError: st.error("Connection Error: Could not connect to the backend."); return False
    except json.JSONDecodeError: st.error("Received an invalid response from the server."); return False
    except Exception as e: st.error(f"An unexpected error during login: {e}"); return False

def register(username, email, password):
    try:
        user_data = {"username": username, "email": email, "password": password}
        response = requests.post(f"{BACKEND_URL}/users/", json=user_data)
        if response.status_code == 200:
            st.success("Registration successful! Please login."); st.session_state.page = "login"; st.rerun(); return True
        else:
            try: detail = response.json().get("detail", response.text)
            except json.JSONDecodeError: detail = response.text
            st.error(f"Registration failed: {detail}"); return False
    except requests.exceptions.ConnectionError: st.error("Connection Error: Could not connect to the backend."); return False
    except Exception as e: st.error(f"An unexpected error during registration: {e}"); return False

def logout():
    st.session_state.token = None; st.session_state.user = None; st.session_state.page = "login"; st.rerun()

# --- Helper functions for API calls ---
def get_user_datasets():
    if "token" not in st.session_state or st.session_state.token is None:
        st.session_state.datasets = []; return False
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        response = requests.get(f"{BACKEND_URL}/users/me/datasets/", headers=headers)
        if response.status_code == 200: st.session_state.datasets = response.json(); return True
        else: st.session_state.datasets = []; return False # Error handled by UI if needed
    except Exception: st.session_state.datasets = []; return False # Error handled by UI if needed

# --- UI Rendering ---
if st.session_state.token is None:
    st.sidebar.title("MochiRAG")
    auth_page_choice = st.sidebar.radio("Choose Action", ["Login", "Register"], index=0 if st.session_state.page == "login" else 1)
    if auth_page_choice == "Login":
        st.session_state.page = "login"; st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username"); password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if not username or not password: st.warning("Please enter both username and password.")
                else: login(username, password)
    elif auth_page_choice == "Register":
        st.session_state.page = "register"; st.header("Register")
        with st.form("register_form"):
            reg_username = st.text_input("Username"); reg_email = st.text_input("Email"); reg_password = st.text_input("Password", type="password")
            if st.form_submit_button("Register"):
                if not reg_username or not reg_email or not reg_password: st.warning("Please fill all fields.")
                else: register(reg_username, reg_email, reg_password)
else:
    st.sidebar.title("MochiRAG Menu")
    if st.session_state.user: st.sidebar.write(f"Welcome, {st.session_state.user.get('username', 'User')}!")
    st.session_state.main_app_page = st.sidebar.radio("Navigate", ["Chat", "Document Management"], key="main_nav")
    if st.sidebar.button("Logout"): logout()

    if st.session_state.main_app_page == "Chat":
        st.title("Chat Page")
        st.sidebar.subheader("RAG Strategy")
        selected_strategy = st.sidebar.selectbox("Choose a RAG strategy:", options=AVAILABLE_RAG_STRATEGIES,
                                                 index=AVAILABLE_RAG_STRATEGIES.index("basic") if "basic" in AVAILABLE_RAG_STRATEGIES else 0,
                                                 key="rag_strategy_selector")
        st.sidebar.caption(f"Current strategy: **{selected_strategy}**")
        if "show_references" not in st.session_state: st.session_state.show_references = False
        st.session_state.show_references = st.sidebar.checkbox("Show references/sources", value=st.session_state.show_references, key="show_references_checkbox")

        st.sidebar.subheader("Target Datasets for Chat")
        # Force initialization if attribute doesn't exist, right before use
        if not hasattr(st.session_state, 'datasets'):
            st.session_state.datasets = []

        if not st.session_state.datasets: # Now this check should be safe
            get_user_datasets()

        current_datasets = getattr(st.session_state, 'datasets', [])
        dataset_options = {ds['name']: ds['dataset_id'] for ds in current_datasets if isinstance(ds, dict) and 'name' in ds and 'dataset_id' in ds}

        selected_dataset_names = st.sidebar.multiselect("Select datasets to query (default: all):", options=list(dataset_options.keys()), key="chat_dataset_selector")
        selected_dataset_ids_for_query = [dataset_options[name] for name in selected_dataset_names if name in dataset_options]

        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        if "chat_loading" not in st.session_state: st.session_state.chat_loading = False
        for msg in st.session_state.chat_history:
            if msg["role"] == "user": st.markdown(f"**You:** {msg['content']}")
            else:
                ai_text = f"<span style='color: #2b7cff'><b>AI</b>{' (<i>'+msg.get('strategy_used','')+')</i>' if msg.get('strategy_used') else ''}:</span> {msg['content']}"
                st.markdown(ai_text, unsafe_allow_html=True)
                if st.session_state.show_references and msg.get("sources"):
                    with st.expander("View Sources", expanded=False):
                        for i, src in enumerate(msg["sources"]):
                            src_text = f"**Source {i+1}:**"; meta = src.get("metadata", {})
                            if meta.get("original_filename"): src_text += f" `{meta['original_filename']}`"
                            elif meta.get("data_source_id"): src_text += f" ID: `{meta['data_source_id']}`"
                            if meta.get("page") is not None: src_text += f" (Page: {meta['page'] + 1})"
                            st.markdown(src_text, unsafe_allow_html=True)
                            if src.get("page_content","")[:150]: st.caption(f"> {src['page_content'][:150]}...")
                        st.markdown("---")
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Your question", key="chat_input", height=80)
            if st.form_submit_button("Send", disabled=st.session_state.chat_loading):
                if user_input.strip():
                    st.session_state.chat_loading = True
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    try:
                        headers = {"Authorization": f"Bearer {st.session_state.token}"}
                        payload = {"question": user_input, "rag_strategy": selected_strategy}
                        if selected_dataset_ids_for_query: payload["dataset_ids"] = selected_dataset_ids_for_query
                        resp = requests.post(f"{BACKEND_URL}/chat/query/", json=payload, headers=headers, timeout=60)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.chat_history.append({"role": "assistant", "content": data.get("answer"), "strategy_used": data.get("strategy_used"), "sources": data.get("sources")})
                        else:
                            detail = resp.text;
                            try: detail = resp.json().get("detail", resp.text)
                            except: pass
                            st.session_state.chat_history.append({"role": "assistant", "content": f"(Error: {detail})", "strategy_used": selected_strategy})
                    except Exception as e: st.session_state.chat_history.append({"role": "assistant", "content": f"(Error: {e})", "strategy_used": selected_strategy})
                    finally: st.session_state.chat_loading = False; st.rerun()
        if st.session_state.chat_loading: st.info(f"AI is thinking (using {selected_strategy} strategy)...")
        if st.button("Clear Chat History"): st.session_state.chat_history = []; st.rerun()

    elif st.session_state.main_app_page == "Document Management":
        st.title("Document Management")

        def create_new_dataset(name: str, description = None):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            payload = {"name": name, "description": description or ""}
            try:
                response = requests.post(f"{BACKEND_URL}/users/me/datasets/", headers=headers, json=payload)
                if response.status_code == 201: st.success(f"Dataset '{name}' created!"); get_user_datasets(); return True
                else: st.error(f"Failed to create dataset: {response.status_code} - {response.text}"); return False
            except Exception as e: st.error(f"Error creating dataset: {e}"); return False

        def delete_selected_dataset(dataset_id_to_delete: str):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                response = requests.delete(f"{BACKEND_URL}/users/me/datasets/{dataset_id_to_delete}/", headers=headers)
                if response.status_code == 204: st.success(f"Dataset deleted!"); st.session_state.selected_dataset_id=None; st.session_state.files_in_selected_dataset=[]; get_user_datasets(); return True
                else: st.error(f"Failed to delete dataset: {response.status_code} - {response.text}"); return False
            except Exception as e: st.error(f"Error deleting dataset: {e}"); return False

        st.subheader("Your Datasets")
        with st.expander("Create New Dataset", expanded=False):
            with st.form("create_dataset_form", clear_on_submit=True):
                new_ds_name = st.text_input("Dataset Name*",key="new_ds_name")
                new_ds_desc = st.text_area("Description",key="new_ds_desc")
                if st.form_submit_button("Create Dataset"):
                    if not new_ds_name.strip(): st.warning("Dataset Name is required.")
                    else: create_new_dataset(new_ds_name, new_ds_desc); st.rerun()

        current_datasets_dm = getattr(st.session_state, 'datasets', [])
        if not current_datasets_dm: get_user_datasets(); current_datasets_dm = getattr(st.session_state, 'datasets', [])

        if current_datasets_dm:
            st.write("Available Datasets:")
            c1,c2,c3,c4=st.columns((2,3,2,2)); c1.write("**Name**");c2.write("**Desc**");c3.write("**Manage**");c4.write("**Del**")
            for ds in current_datasets_dm:
                if not isinstance(ds, dict): continue
                col1,col2,col3,col4=st.columns((2,3,2,2))
                col1.write(ds.get("name")); col2.write(ds.get("description")or"");
                if col3.button("Files",key=f"mng_{ds.get('dataset_id')}"):st.session_state.selected_dataset_id=ds.get("dataset_id");st.session_state.files_in_selected_dataset=[];st.rerun()
                if col4.button("🗑️",key=f"del_ds_{ds.get('dataset_id')}"):
                    st.session_state.deleting_dataset_id = ds.get('dataset_id')
            if "deleting_dataset_id" in st.session_state and st.session_state.deleting_dataset_id:
                ds_to_delete = next((d for d in current_datasets_dm if isinstance(d, dict) and d.get('dataset_id') == st.session_state.deleting_dataset_id), None)
                if ds_to_delete:
                    st.warning(f"Delete dataset '{ds_to_delete.get('name')}' and all its files?")
                    if st.button("Confirm Delete",key=f"conf_del_{ds_to_delete.get('dataset_id')}"): delete_selected_dataset(ds_to_delete.get('dataset_id')); st.session_state.deleting_dataset_id=None; st.rerun()
                    if st.button("Cancel",key=f"cancel_del_{ds_to_delete.get('dataset_id')}"): st.session_state.deleting_dataset_id=None; st.rerun()
        else: st.info("No datasets yet.")
        st.markdown("---")

        if st.session_state.selected_dataset_id:
            sel_ds = next((d for d in current_datasets_dm if isinstance(d,dict) and d.get('dataset_id')==st.session_state.selected_dataset_id),None)
            if not sel_ds: st.error("Selected dataset error."); st.session_state.selected_dataset_id=None; st.rerun()
            else: st.subheader(f"Files in: \"{sel_ds.get('name')}\"")
            def get_files():
                h={"Authorization":f"Bearer {st.session_state.token}"}
                try:
                    r=requests.get(f"{BACKEND_URL}/users/me/datasets/{st.session_state.selected_dataset_id}/documents/",headers=h)
                    if r.status_code==200: st.session_state.files_in_selected_dataset=r.json()
                    else: st.error(f"Failed to fetch files: {r.status_code}-{r.text}"); st.session_state.files_in_selected_dataset=[]
                except Exception as e:st.error(f"Error fetching files:{e}");st.session_state.files_in_selected_dataset=[]
            def del_file(d_id:str):
                h={"Authorization":f"Bearer {st.session_state.token}"}
                try:
                    r=requests.delete(f"{BACKEND_URL}/users/me/datasets/{st.session_state.selected_dataset_id}/documents/{d_id}/",headers=h)
                    if r.status_code==204:st.success("File deleted!");get_files();return True
                    else:st.error(f"Failed to delete file:{r.status_code}-{r.text}");return False
                except Exception as e:st.error(f"Error deleting file:{e}");return False

            current_files = getattr(st.session_state, 'files_in_selected_dataset', [])
            if not current_files and st.session_state.selected_dataset_id:get_files(); current_files = getattr(st.session_state, 'files_in_selected_dataset', [])

            st.markdown("#### Upload New Document")
            emb_strats=["default"]; chk_strats=["default"]

            local_embedding_manager = globals().get('embedding_manager')
            if local_embedding_manager is not None:
                try: emb_strats = local_embedding_manager.get_available_strategies()
                except Exception as e: st.error(f"Failed to load emb strats: {e}")
            elif embedding_manager is None and 'st' in globals(): # Check if st is available before using
                 st.warning("Embedding manager not available. Using default.")

            ul_emb_strat=st.selectbox("Emb Strat:",options=emb_strats,key="ul_emb")

            local_chunking_manager = globals().get('chunking_manager')
            if local_chunking_manager is not None:
                try: chk_strats = local_chunking_manager.get_available_strategies()
                except Exception as e: st.error(f"Failed to load chk strats: {e}")
            elif chunking_manager is None and 'st' in globals():
                 st.warning("Chunking manager not available. Using default.")
            ul_chk_strat=st.selectbox("Chk Strat:",options=chk_strats,key="ul_chk")
            ul_cs=st.number_input("Chunk Size",value=1000,min_value=100,step=50,key="ul_cs_val")
            ul_co=st.number_input("Chunk Overlap",value=200,min_value=0,step=20,key="ul_co_val")
            with st.form("ul_file_ds_form",clear_on_submit=True):
                uploaded_files = st.file_uploader("Upload Document(s) (TXT,MD,PDF)",type=["txt","md","pdf"],key="f_ul_ds", accept_multiple_files=True)
                if st.form_submit_button("Upload to Dataset"):
                    if uploaded_files and st.session_state.selected_dataset_id:
                        success_count = 0
                        error_count = 0
                        for ul_f in uploaded_files:
                            pl={"emb_strat":ul_emb_strat if ul_emb_strat!="default"else None,"chk_strat":ul_chk_strat if ul_chk_strat!="default"else None}
                            if ul_chk_strat and"recursive"in ul_chk_strat.lower():pl["chk_params"]={"chunk_size":ul_cs,"chunk_overlap":ul_co}
                            fd_pl={"embedding_strategy":pl["emb_strat"],"chunking_strategy":pl["chk_strat"]}
                            if pl.get("chk_params"):fd_pl['chunking_params_json']=json.dumps(pl["chk_params"])
                            f_pl={"file":(ul_f.name,ul_f,ul_f.type)};h_auth={"Authorization":f"Bearer {st.session_state.token}"}
                            try:
                                rp=requests.post(f"{BACKEND_URL}/users/me/datasets/{st.session_state.selected_dataset_id}/documents/upload/",headers=h_auth,files=f_pl,data=fd_pl,timeout=120)
                                if rp.status_code==200:
                                    success_count += 1
                                else:
                                    st.error(f"Upload fail for '{ul_f.name}': {rp.status_code}-{rp.text}")
                                    error_count += 1
                            except Exception as e:
                                st.error(f"Upload err for '{ul_f.name}': {e}")
                                error_count += 1

                        if success_count > 0:
                            st.success(f"{success_count} file(s) uploaded successfully.")
                            get_files()
                        if error_count > 0:
                            st.warning(f"{error_count} file(s) failed to upload.")
                        st.rerun()
            st.markdown("---");st.markdown("#### Documents in Dataset")
            if st.button("Refresh Files",key="ref_files"):get_files();st.rerun()

            current_files_display = getattr(st.session_state, 'files_in_selected_dataset', [])
            if current_files_display:
                c1f,c2f,c3f,c4f=st.columns((3,2,2,1));c1f.write("**File**");c2f.write("**Date**");c3f.write("**Chunks**");c4f.write("**Del**")
                for fm in current_files_display:
                    if not isinstance(fm, dict): continue
                    fc1,fc2,fc3,fc4=st.columns((3,2,2,1));fc1.write(fm.get("original_filename"));fc2.write(fm.get("uploaded_at","").split("T")[0]);fc3.write(str(fm.get("chunk_count","-")));
                    if fc4.button("🗑️",key=f"delf_{fm.get('data_source_id')}",help="Delete file"):
                        st.session_state.deleting_file_id = fm.get('data_source_id')
                        st.session_state.deleting_file_name = fm.get('original_filename')
                if "deleting_file_id" in st.session_state and st.session_state.deleting_file_id:
                    st.warning(f"Delete file '{st.session_state.deleting_file_name}'?")
                    if st.button("Confirm Del File",key=f"conf_delf_{st.session_state.deleting_file_id}"):del_file(st.session_state.deleting_file_id);st.session_state.deleting_file_id=None;st.session_state.deleting_file_name=None;st.rerun()
                    if st.button("Cancel Del File",key=f"cancel_delf_{st.session_state.deleting_file_id}"):st.session_state.deleting_file_id=None;st.session_state.deleting_file_name=None;st.rerun()
            else:st.info("No docs in this dataset.")
        else:st.info("Select dataset to manage files.")
    else:
        st.title("Welcome")
        st.write("Select a page from the sidebar.")

# To run this app:
# 1. Ensure the FastAPI backend is running (e.g., uvicorn backend.main:app --reload --port 8000)
# 2. Install frontend requirements: pip install -r frontend/requirements.txt
# 3. Run Streamlit: streamlit run frontend/app.py
