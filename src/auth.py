import jwt
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


def generate_dummy_token():
    payload = {
        "sub": "dummy",
        "exp": datetime.utcnow() + timedelta(days=365),
        "scope": "internal",
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


dummy_token = generate_dummy_token()
print(dummy_token)


security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") != "dummy":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid dummy token"
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid dummy token"
        )


if __name__ == "__main__":
    token = generate_dummy_token()
