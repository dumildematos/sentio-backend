import asyncio
import logging
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import settings
from services.session_manager import session_manager

router = APIRouter()
logger = logging.getLogger("sentio.websocket")


def _is_allowed_origin(origin: str | None) -> bool:
    if origin is None:
        return True

    if origin in settings.cors_allowed_origins:
        return True

    return re.fullmatch(settings.cors_allowed_origin_regex, origin) is not None


@router.websocket(settings.ws_endpoint)
async def brain_stream(websocket: WebSocket):
    """
    Streams the latest background EEG message to the frontend.
    """
    origin = websocket.headers.get("origin")
    if not _is_allowed_origin(origin):
        logger.warning("Rejected websocket connection from origin %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    last_timestamp = None

    try:
        while True:
            message = session_manager.get_latest_stream_message()
            if message is None:
                if not session_manager.is_active() and not session_manager.is_streaming():
                    break
                await asyncio.sleep(0.1)
                continue

            timestamp = message.get("timestamp")
            if timestamp != last_timestamp:
                await websocket.send_json(message)
                last_timestamp = timestamp

            await asyncio.sleep(settings.eeg_update_interval)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()