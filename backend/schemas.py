from pydantic import BaseModel, EmailStr

# --- User Schemas ---

# Properties to receive via API on user creation
class UserCreate(BaseModel):
    email: EmailStr
    password: str

# Properties to return via API, hiding sensitive data like password
class User(BaseModel):
    id: int
    email: EmailStr
    is_active: bool

    class Config:
        # This allows the model to be created from an ORM object (like our User model)
        orm_mode = True
