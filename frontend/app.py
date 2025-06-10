import streamlit as st
import requests
import json # For handling potential JSON decode errors more gracefully

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
        st.write("Chat interface will be here.")
        # Placeholder for chat functionality

    elif st.session_state.main_app_page == "Document Management":
        st.title("Document Management")
        st.write("Document upload and listing will be here.")
        # Placeholder for document management UI

    else:
        st.title("Welcome")
        st.write("Select a page from the sidebar.")

# To run this app:
# 1. Ensure the FastAPI backend is running (e.g., uvicorn backend.main:app --reload --port 8000)
# 2. Install frontend requirements: pip install -r frontend/requirements.txt
# 3. Run Streamlit: streamlit run frontend/app.py
