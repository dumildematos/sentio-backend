import asyncio
import logging
import re
import time

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


def _get_client_identity(websocket: WebSocket) -> tuple[str, int | str]:
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    return client_host, client_port


def _log_connection_requested(client_host: str, client_port: int | str, origin: str | None):
    logger.info(
        "WebSocket connection requested for %s from %s:%s origin=%s",
        settings.ws_endpoint,
        client_host,
        client_port,
        origin,
    )


def _log_connection_accepted(client_host: str, client_port: int | str):
    logger.info(
        "WebSocket accepted for %s from %s:%s active=%s streaming=%s",
        settings.ws_endpoint,
        client_host,
        client_port,
        session_manager.is_active(),
        session_manager.is_streaming(),
    )


def _log_waiting_for_data(client_host: str, client_port: int | str, last_wait_log: float) -> float:
    now = time.monotonic()
    if now - last_wait_log >= 5.0:
        logger.info(
            "WebSocket waiting for EEG data for %s:%s active=%s streaming=%s",
            client_host,
            client_port,
            session_manager.is_active(),
            session_manager.is_streaming(),
        )
        return now

    return last_wait_log


def _should_close_idle_socket(client_host: str, client_port: int | str) -> bool:
    if session_manager.is_active() or session_manager.is_streaming():
        return False

    logger.info(
        "Closing WebSocket for %s:%s because session is inactive and no stream is running",
        client_host,
        client_port,
    )
    return True


async def _send_new_message(websocket: WebSocket, message: dict, last_timestamp, messages_sent: int, client_host: str, client_port: int | str) -> tuple[object, int]:
    timestamp = message.get("timestamp")
    if timestamp == last_timestamp:
        return last_timestamp, messages_sent

    await websocket.send_json(message)
    messages_sent += 1

    if messages_sent == 1 or messages_sent % 25 == 0:
        logger.info(
            "Sent EEG frame %s to %s:%s timestamp=%s emotion=%s",
            messages_sent,
            client_host,
            client_port,
            timestamp,
            message.get("emotion"),
        )

    return timestamp, messages_sent


@router.websocket(settings.ws_endpoint)
async def brain_stream(websocket: WebSocket):
    """
    Streams the latest background EEG message to the frontend.
    """
    origin = websocket.headers.get("origin")
    client_host, client_port = _get_client_identity(websocket)
    _log_connection_requested(client_host, client_port, origin)

    if not _is_allowed_origin(origin):
        logger.warning("Rejected websocket connection from origin %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    last_timestamp = None
    messages_sent = 0
    last_wait_log = 0.0
    _log_connection_accepted(client_host, client_port)

    try:
        while True:
            message = session_manager.get_latest_stream_message()
            if message is None:
                if _should_close_idle_socket(client_host, client_port):
                    break

                last_wait_log = _log_waiting_for_data(client_host, client_port, last_wait_log)
                await asyncio.sleep(0.1)
                continue

            last_timestamp, messages_sent = await _send_new_message(
                websocket,
                message,
                last_timestamp,
                messages_sent,
                client_host,
                client_port,
            )

            await asyncio.sleep(settings.eeg_update_interval)
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected for %s:%s after %s frames",
            client_host,
            client_port,
            messages_sent,
        )
    except Exception:
        logger.exception(
            "WebSocket error for %s:%s after %s frames",
            client_host,
            client_port,
            messages_sent,
        )
        await websocket.close()
    else:
        logger.info(
            "WebSocket loop ended for %s:%s after %s frames",
            client_host,
            client_port,
            messages_sent,
        )