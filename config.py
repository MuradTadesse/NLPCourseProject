# config/config.py
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "allowed_origins": ["https://chat.muradtadesse.com"]
}

# api/app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["allowed_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)