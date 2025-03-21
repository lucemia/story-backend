from fastapi import APIRouter

from app.api.headless import router as headless_router

router = APIRouter()


@router.get("/ok")
async def ok():
    return {"ok": True}


router.include_router(headless_router, prefix="/headless", tags=["headless"])
