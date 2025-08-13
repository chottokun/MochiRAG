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
def render_auth_page():
    st.title("Welcome to MochiRAG")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.subheader("Login")
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if st.session_state.api_client.login(email, password):
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    with signup_tab:
        st.subheader("Create a New Account")
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if not email or not password:
                    st.error("Email and password are required.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    if st.session_state.api_client.signup(email, password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Failed to create account. The email might already be in use.")

def main():
    initialize_session_state()

    if not st.session_state.api_client.token:
        render_auth_page()
    else:
        # If logged in, show the main application
        render_sidebar()
        render_chat()

if __name__ == "__main__":
    main()
