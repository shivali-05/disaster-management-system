"""FastAPI entry point for the project demo."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:  # ``uvicorn backend.main:app`` from the repository root
    from .api.routes import router
except ImportError:  # ``cd backend && uvicorn main:app --reload``
    from api.routes import router

app = FastAPI(title="Multi-Hazard Risk Assessment API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
