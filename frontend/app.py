import streamlit as st
from api_client import ApiClient
from ui_sidebar import render_sidebar
from ui_chat import render_chat

# --- Page Configuration ---
st.set_page_config(page_title="MochiRAG", layout="wide")

# --- State Management ---
def initialize_session_state():
    if "api_client" not in st.session_state:
        st.session_state.api_client = ApiClient(base_url="http://localhost:8000")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "datasets" not in st.session_state:
        st.session_state.datasets = []

# --- Authentication and Main App --- 
def main():
    initialize_session_state()

    # If user is not logged in, show login form
    if not st.session_state.api_client.token:
        st.title("Login to MochiRAG")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if st.session_state.api_client.login(email, password):
                    st.rerun()
                else:
                    st.error("Invalid email or password")
    else:
        # If logged in, show the main application
        render_sidebar()
        render_chat()

if __name__ == "__main__":
    main()
