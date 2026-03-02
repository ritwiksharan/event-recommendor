import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import recommend, qa

app = FastAPI(title="EventScout API", version="1.0.0")

# ── API routes ─────────────────────────────────────────────────────────────────
app.include_router(recommend.router, prefix="/api")
app.include_router(qa.router,        prefix="/api")

# ── Serve frontend static files ────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
