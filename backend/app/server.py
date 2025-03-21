from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.api import router as api_router


app = FastAPI(title="OpenGPTs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get root of app, used to point to directory containing static files
ROOT = Path(__file__).parent.parent


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    uvicorn.run(app, host="0.0.0.0", port=8100)
