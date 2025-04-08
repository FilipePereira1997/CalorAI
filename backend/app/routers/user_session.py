# user_session.py
import uuid
import logging
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

# Set up logger
logger = logging.getLogger("user_session")
logger.setLevel(logging.INFO)

router = APIRouter()

# Global dictionary to simulate session storage (not recommended for production use)
user_sessions = {}

@router.post("/session")
async def create_session(request: Request, response: Response):
    """
    Receives user data (e.g., weight, height, age, gender, activity level) and creates a session.
    Returns a unique session_id and stores the data in the session.
    """
    data = await request.json()
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = data
    logger.info(f"Session created: {session_id} with data: {data}")
    res = JSONResponse(content={"message": "Session created", "session_id": session_id})
    res.set_cookie(key="session_id", value=session_id)
    return res

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Retrieves session data based on the provided session_id.
    """
    session_data = user_sessions.get(session_id)
    if not session_data:
        logger.warning(f"Session not found: {session_id}")
        return JSONResponse(content={"message": "Session not found"}, status_code=404)
    logger.info(f"Session retrieved: {session_id} with data: {session_data}")
    return {"session_id": session_id, "data": session_data}

@router.get("/sessions")
async def get_all_sessions():
    """
    Retrieves all session data.
    """
    logger.info(f"Retrieving all sessions. Total sessions: {len(user_sessions)}")
    return {"total_sessions": len(user_sessions), "sessions": user_sessions}
