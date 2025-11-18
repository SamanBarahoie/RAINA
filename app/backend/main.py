from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .routers import memory, sessions, knowledge, chat

app = FastAPI(title="RAINA API", version="1.0")

# ======================================
# CORS settings for Streamlit on localhost
# ======================================
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],       # Allow all methods in DEV environment
    allow_headers=["*"],       # Allow all headers in DEV environment
)

# ======================================
# Security headers for basic protection (safe for DEV)
# ======================================
@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add basic security headers to all responses."""
    response = await call_next(request)

    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=()"
    )

    # Uncomment for production (HTTPS required)
    # response.headers["Strict-Transport-Security"] = (
    #     "max-age=63072000; includeSubDomains; preload"
    # )

    return response


# ======================================
# Include all routers
# ======================================
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(knowledge.router)
app.include_router(memory.router)


# ======================================
# Health check endpoint
# ======================================
@app.get("/")
def root():
    """Return service status."""
    return {
        "status": "ok",
        "message": "RAG Chatbot backend is running (DEV mode)",
    }
