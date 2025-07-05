import pytest
import frontend.app as app
from unittest.mock import patch, MagicMock

# login関数の正常系
@patch("frontend.app.requests.post")
@patch("frontend.app.requests.get")
def test_login_success(mock_get, mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "tok"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"username": "u", "email": "e"}
    # session_stateをモック
    app.st.session_state.token = None
    app.st.session_state.user = None
    app.st.session_state.page = "login"
    result = app.login("u", "p")
    assert result is True
    assert app.st.session_state.token == "tok"
    assert app.st.session_state.user["username"] == "u"
    assert app.st.session_state.page == "app_main"

# login関数の失敗系
@patch("frontend.app.requests.post")
def test_login_fail(mock_post):
    mock_post.return_value.status_code = 401
    mock_post.return_value.text = "fail"
    app.st.session_state.token = None
    result = app.login("u", "bad")
    assert result is False
    assert app.st.session_state.token is None

# register関数の正常系
@patch("frontend.app.requests.post")
def test_register_success(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"username": "u", "email": "e"}
    app.st.session_state.page = "register"
    result = app.register("u", "e", "p")
    assert result is True
    assert app.st.session_state.page == "login"

# register関数の失敗系
@patch("frontend.app.requests.post")
def test_register_fail(mock_post):
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {"detail": "fail"}
    app.st.session_state.page = "register"
    result = app.register("u", "e", "p")
    assert result is False
    assert app.st.session_state.page == "register"

# logout関数

def test_logout():
    app.st.session_state.token = "tok"
    app.st.session_state.user = {"username": "u"}
    app.st.session_state.page = "app_main"
    app.logout()
    assert app.st.session_state.token is None
    assert app.st.session_state.user is None
    assert app.st.session_state.page == "login"
