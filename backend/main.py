import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

from core import UltraPersonalizationEngine

app = FastAPI(title="Chameleon Personalization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[UltraPersonalizationEngine] = None

ASSET_DIR = os.environ.get("CHAMELEON_ASSET_DIR", "chameleon_api_assets")
CATALOG_PATH = os.environ.get("CHAMELEON_CATALOG_PATH", "data/catalog.json")


class UserPayload(BaseModel):
    user_pseudo_id: Optional[str] = None
    total_events: int = 1
    unique_pages: int = 1
    total_revenue: float = 0.0
    days_since_first_visit: int = 0
    recent_sequence: List[str] = []
    recent_items: List[str] = []
    recent_categories: List[str] = []
    recent_display_categories: List[str] = []
    # Demographic/context signals: the current frontend doesn't collect
    # these yet, so they default to "unknown" and the cold-start resolver
    # backs off gracefully (see ml_models.ColdStartResolver). Wired up here
    # so the API is ready the moment the frontend starts sending them.
    region: Optional[str] = None
    country: Optional[str] = None
    source: Optional[str] = None
    medium: Optional[str] = None
    Age: Optional[str] = None
    category: Optional[str] = None  # device category (mobile/desktop/tablet)


def guess_device_category(user_agent: str) -> str:
    """
    Cheap, real cold-start signal we can get for free without any frontend
    change: infer device category from the request's User-Agent header.
    Only used when the payload doesn't already specify `category`.
    """
    if not user_agent:
        return "unknown"
    ua = user_agent.lower()
    if "tablet" in ua or "ipad" in ua:
        return "tablet"
    if "mobi" in ua or "android" in ua or "iphone" in ua:
        return "mobile"
    return "desktop"


@app.on_event("startup")
async def load_models():
    global engine
    logger.info("Booting up API and loading ML models into memory...")

    try:
        engine = UltraPersonalizationEngine.load_for_inference(
            asset_dir=ASSET_DIR,
            catalog_path=CATALOG_PATH,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        logger.info("Models successfully loaded from disk. Ready for traffic.")
    except Exception as e:
        logger.exception(f"Critical error loading models: {e}")
        engine = None


@app.get("/api/v1/health")
async def health():
    ready = engine is not None and getattr(engine.pm, "kmeans", None) is not None
    return {"status": "ok" if ready else "loading", "catalog_loaded": bool(engine and engine.catalog)}


@app.post("/api/v1/personalize")
async def personalize_ui(payload: UserPayload, request: Request):
    if engine is None or getattr(engine.pm, "kmeans", None) is None:
        raise HTTPException(status_code=503, detail="Models are still loading or offline...")

    profile_dict = payload.model_dump()

    # Fill in context defaults / free signals not sent by the current
    # frontend. Anything the payload didn't set stays "unknown", which the
    # cold-start resolver handles gracefully via its backoff hierarchy.
    profile_dict["category"] = payload.category or guess_device_category(request.headers.get("user-agent", ""))
    for key in ("region", "country", "source", "medium", "Age"):
        profile_dict[key] = profile_dict.get(key) or "unknown"

    try:
        result = engine.generate_personalized_landing_page(profile_dict)
        return result
    except Exception as e:
        logger.exception("Personalization request failed")
        raise HTTPException(status_code=500, detail=str(e))
