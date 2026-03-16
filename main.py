import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from api.websocket import router as ws_router
from config import settings

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("sentio.api")

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_origin_regex=settings.cors_allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_api_requests(request: Request, call_next):
    if not request.url.path.startswith("/api"):
        return await call_next(request)

    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    query_string = f"?{request.url.query}" if request.url.query else ""

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "%s %s%s -> unhandled error %.2fms [%s]",
            request.method,
            request.url.path,
            query_string,
            duration_ms,
            client_host,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "%s %s%s -> %s %.2fms [%s]",
        request.method,
        request.url.path,
        query_string,
        response.status_code,
        duration_ms,
        client_host,
    )
    return response

# -----------------------------
# Include routers
# -----------------------------
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Sentio EEG Backend is running."}


# Run with either:
#   python main.py
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )