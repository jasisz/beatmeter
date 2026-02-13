"""FastAPI application - serves API and frontend."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from beatmeter.api.upload import router as upload_router
from beatmeter.api.websocket import router as ws_router

app = FastAPI(title="Beatmeter", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(ws_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
FRONTEND_ROOT = FRONTEND_DIR.resolve() if FRONTEND_DIR.exists() else None

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    def _safe_frontend_file(path: str) -> Path | None:
        if FRONTEND_ROOT is None:
            return None
        candidate = (FRONTEND_ROOT / path.lstrip("/")).resolve()
        if candidate != FRONTEND_ROOT and FRONTEND_ROOT not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    @app.get("/")
    async def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{path:path}")
    async def catch_all(path: str):
        file_path = _safe_frontend_file(path)
        if file_path is not None:
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))


def run():
    import uvicorn
    from beatmeter.config import settings
    uvicorn.run(
        "beatmeter.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
