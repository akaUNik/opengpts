from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

from fastapi import APIRouter, HTTPException, Path
from langchain.schema.messages import AnyMessage
from pydantic import BaseModel, Field

import app.storage as storage
from app.auth.handlers import AuthedUser
from app.schema import Thread

router = APIRouter()


ThreadID = Annotated[str, Path(description="The ID of the thread.")]


class ThreadPutRequest(BaseModel):
    """Payload for creating a thread."""

    name: str = Field(..., description="The name of the thread.")
    assistant_id: str = Field(..., description="The ID of the assistant to use.")


class ThreadPostRequest(BaseModel):
    """Payload for adding state to a thread."""

    values: Union[Sequence[AnyMessage], Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None


class ThreadPatchRequest(BaseModel):
    """Payload for patching thread state."""

    metadata: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


@router.get("/")
async def list_threads(user: AuthedUser) -> List[Thread]:
    """List all threads for the current user."""
    return await storage.list_threads(user["user_id"])


@router.get("/{tid}/state")
async def get_thread_state(
    user: AuthedUser,
    tid: ThreadID,
):
    """Get state for a thread."""
    thread = await storage.get_thread(user["user_id"], tid)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    assistant = await storage.get_assistant(user["user_id"], thread["assistant_id"])
    if not assistant:
        raise HTTPException(status_code=400, detail="Thread has no assistant")
    return await storage.get_thread_state(
        user_id=user["user_id"],
        thread_id=tid,
        assistant=assistant,
    )


@router.post("/{tid}/state")
async def add_thread_state(
    user: AuthedUser,
    tid: ThreadID,
    payload: ThreadPostRequest,
):
    """Add state to a thread."""
    thread = await storage.get_thread(user["user_id"], tid)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    assistant = await storage.get_assistant(user["user_id"], thread["assistant_id"])
    if not assistant:
        raise HTTPException(status_code=400, detail="Thread has no assistant")
    return await storage.update_thread_state(
        payload.config or {"configurable": {"thread_id": tid}},
        payload.values,
        user_id=user["user_id"],
        assistant=assistant,
    )


@router.patch("/{tid}/state")
async def patch_thread_state(
    user: AuthedUser,
    tid: ThreadID,
    payload: ThreadPatchRequest,
):
    """Patch state for a thread."""
    thread = await storage.get_thread(user["user_id"], tid)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    return await storage.patch_thread_state(
        payload.config or {"configurable": {"thread_id": tid}},
        payload.metadata,
    )


@router.get("/{tid}/history")
async def get_thread_history(
    user: AuthedUser,
    tid: ThreadID,
):
    """Get all past states for a thread."""
    thread = await storage.get_thread(user["user_id"], tid)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    assistant = await storage.get_assistant(user["user_id"], thread["assistant_id"])
    if not assistant:
        raise HTTPException(status_code=400, detail="Thread has no assistant")
    return await storage.get_thread_history(
        user_id=user["user_id"],
        thread_id=tid,
        assistant=assistant,
    )


@router.get("/{tid}")
async def get_thread(
    user: AuthedUser,
    tid: ThreadID,
) -> Thread:
    """Get a thread by ID."""
    thread = await storage.get_thread(user["user_id"], tid)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.post("")
async def create_thread(
    user: AuthedUser,
    thread_put_request: ThreadPutRequest,
) -> Thread:
    """Create a thread."""
    return await storage.create_thread(
        user["user_id"],
        assistant_id=thread_put_request.assistant_id,
        name=thread_put_request.name,
    )


@router.patch("/{tid}")
async def patch_thread(
    user: AuthedUser,
    tid: ThreadID,
    thread_put_request: ThreadPutRequest,
) -> Thread:
    """Update a thread."""
    return await storage.patch_thread(
        user["user_id"],
        tid,
        assistant_id=thread_put_request.assistant_id,
        name=thread_put_request.name,
    )


@router.delete("/{tid}")
async def delete_thread(
    user: AuthedUser,
    tid: ThreadID,
):
    """Delete a thread by ID."""
    await storage.delete_thread(user["user_id"], tid)
    return {"status": "ok"}
