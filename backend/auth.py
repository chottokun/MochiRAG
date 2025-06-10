import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import EmailStr, ValidationError

from .models import UserCreate, UserInDB, TokenData, User

# Configuration for passlib
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Path to the users JSON database
USERS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "users.json"

# JWT Configuration
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def _ensure_users_db_exists():
    """Ensures the users.json file exists and is initialized with an empty list if not."""
    if not USERS_DB_PATH.exists():
        USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_DB_PATH, "w") as f:
            json.dump([], f)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return PWD_CONTEXT.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a password."""
    return PWD_CONTEXT.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """
    Reads users.json, finds a user by username, and returns a UserInDB instance or None.
    """
    _ensure_users_db_exists()
    if not USERS_DB_PATH.is_file(): # Ensure it's a file, not a dir
        return None
    with open(USERS_DB_PATH, "r") as f:
        try:
            users_data = json.load(f)
        except json.JSONDecodeError:
            users_data = []

    for user_dict_str in users_data:
        # The user_id might be stored as str, convert to UUID for UserInDB
        if isinstance(user_dict_str, dict) and user_dict_str.get("username") == username:
            try:
                # Ensure user_id is valid UUID if it's a string
                if 'user_id' in user_dict_str and isinstance(user_dict_str['user_id'], str):
                    user_dict_str['user_id'] = uuid.UUID(user_dict_str['user_id'])
                return UserInDB(**user_dict_str)
            except ValidationError: # Handle cases where data in json is not valid UserInDB
                continue # Or log error, skip user
    return None

def create_user_in_db(user_data: UserCreate) -> UserInDB:
    """
    Creates a new user in the JSON database.
    """
    _ensure_users_db_exists()
    if get_user(user_data.username):
        raise ValueError(f"User with username '{user_data.username}' already exists.")

    try:
        EmailStr.validate(user_data.email)
    except ValueError as e:
        raise ValueError(f"Invalid email format: {user_data.email}. Error: {e}")


    hashed_password = get_password_hash(user_data.password)
    new_user_id = uuid.uuid4()
    user_in_db = UserInDB(
        user_id=new_user_id,
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        disabled=False
    )

    users_list = []
    if USERS_DB_PATH.exists() and USERS_DB_PATH.stat().st_size > 0:
        with open(USERS_DB_PATH, "r") as f:
            try:
                users_list = json.load(f)
            except json.JSONDecodeError:
                users_list = []

    user_dict = user_in_db.model_dump()
    user_dict['user_id'] = str(user_in_db.user_id) # Serialize UUID to string for JSON

    users_list.append(user_dict)
    with open(USERS_DB_PATH, "w") as f: # Open in write mode to overwrite
        json.dump(users_list, f, indent=4)

    return user_in_db

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    # Ensure all data going into JWT is serializable (e.g. UUID to str)
    for key, value in to_encode.items():
        if isinstance(value, uuid.UUID):
            to_encode[key] = str(value)

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user by username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Decodes JWT, validates, and returns the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        user_id_str: Optional[str] = payload.get("user_id")

        if username is None and user_id_str is None: # Ensure we have at least one identifier
            raise credentials_exception

        # Reconstruct TokenData if needed, or directly use fields
        # token_data = TokenData(username=username, user_id=uuid.UUID(user_id_str) if user_id_str else None)

    except JWTError:
        raise credentials_exception

    # Prefer username if available, as get_user uses it
    user_identity = username
    if not user_identity and user_id_str:
        # If only user_id is in token, this implies get_user needs to support it
        # For now, our get_user is by username. So, username must be in token.
        # Let's assume 'sub' will always be the username for now.
        # If user_id was the primary identifier in token, get_user would need adjustment
        # or a new function get_user_by_id would be needed.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token must contain username (sub)",
        )

    user_in_db = get_user(username=user_identity)
    if user_in_db is None:
        raise credentials_exception

    # Return a User model, not UserInDB, to exclude hashed_password
    return User(**user_in_db.model_dump())


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Checks if the current user is active (not disabled)."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Initialize the users.json file if it doesn't exist when this module is loaded.
_ensure_users_db_exists()
