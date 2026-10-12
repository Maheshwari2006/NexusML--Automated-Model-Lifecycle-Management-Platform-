
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.src.pipeline.model_registry import (
    list_versions,
    rollback_model,
)

router = APIRouter(prefix="/models", tags=["Model Versioning"])


class RollbackRequest(BaseModel):
    version: int = Field(ge=1)


@router.get("/versions")
def get_versions():
    return list_versions()


@router.post("/rollback")
def rollback(request: RollbackRequest):
    try:
        registry = rollback_model(request.version)
        return {
            "message": f"Rolled back to v{request.version}",
            "registry": registry,
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
