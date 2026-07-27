"""
Development server entry point.

Usage:
    python run.py
    # or
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
