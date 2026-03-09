import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import settings
from services.session_manager import session_manager

router = APIRouter()


@router.websocket(settings.ws_endpoint)
async def brain_stream(websocket: WebSocket):
    """
    Streams the latest background EEG message to the frontend.
    """
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