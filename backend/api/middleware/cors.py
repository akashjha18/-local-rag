"""
cors.py — CORS Configuration
==============================
Allows the React frontend (localhost:5173) to call
the FastAPI backend (localhost:8000).

CORS = Cross-Origin Resource Sharing.
Browsers block requests between different origins by default.
This middleware tells the browser our frontend is allowed.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_cors(app: FastAPI) -> None:
    """
    Add CORS middleware to the FastAPI app.

    In development: allow all localhost origins.
    In production: restrict to your actual domain.
    """
    app.add_middleware(
        CORSMiddleware,
        # Origins allowed to make requests
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",   # Alternative React port
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],           # GET, POST, DELETE, etc.
        allow_headers=["*"],           # Content-Type, Authorization, etc.
    )