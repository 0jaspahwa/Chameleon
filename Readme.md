# Chameleon

A hyper-personalized landing-page generator for e-commerce. Chameleon assembles a different homepage — hero copy, product selection, CTA — for every visitor, including brand-new anonymous ones, by combining a trained behavioral-segmentation ML stack with a demographic/contextual cold-start resolver, and rendering results through a real product catalog.

Built for the **NetElixir AIgnition 2.0** hackathon.

## What's in this repo

```
chameleon/
├── backend/         FastAPI service + ML training pipeline
├── frontend/        React + TypeScript showcase UI
└── docs/            Model documentation (Word)
```

The backend serves a stateless personalization API. The frontend is a showcase page that lets you switch between five simulated visitor profiles and watch the page — hero, products, tone, telemetry — reassemble live against the real model.

## Why it exists

Most recommendation systems break for first-time or anonymous visitors: they need behavioral history that doesn't exist yet. Chameleon addresses this with two distinct code paths:

- **Behavioral stack** (Random Forest + Deep NN + LSTM) for visitors with enough session signal
- **Cold-start resolver** — a hierarchical demographic/contextual fallback (region → device → source → medium, backing off from most to least specific) for everyone else

The response feeds into a React frontend that renders real catalog products, with an on-screen ML telemetry panel showing the predicted segment, confidence score, and prediction source in real time.

## Results (real dataset)

Held-out validation accuracy on the hackathon's dataset (~24,700 users):

| Model                    | Accuracy | F1 (weighted) |
|--------------------------|---------:|--------------:|
| Majority-class baseline  |   0.5065 |          —    |
| Random Forest            |   0.9779 |     0.9779    |
| Deep Predictor           |   0.9887 |     0.9887    |
| LSTM (sequence)          |   0.9017 |     0.9017    |
| **Deployed cascade**     |**0.9800**|   **0.9800**  |

Per-segment breakdown of the deployed cascade (n=500 validation users):

```
                        precision    recall  f1-score   support
   one_time_buyer_seg0       0.98      0.98      0.98       255
   one_time_buyer_seg1       0.98      0.98      0.98       228
      repeat_purchaser       1.00      0.94      0.97        17
              accuracy                           0.98       500
```

Full accuracy report prints automatically at the end of every training run.

## Architecture

```
data_processor.py    -> loads & cleans raw activity/transaction CSVs
ml_models.py         -> PersonalizationModel (RF + Deep NN + LSTM),
                          ColdStartResolver, SelectiveScaler
core.py              -> UltraPersonalizationEngine: training pipeline
                          + stateless inference logic
catalog_service.py   -> maps ML segments to REAL catalog products
run_pipeline.py      -> trains everything, exports model assets
main.py              -> FastAPI server, loads assets, serves requests
```

Two design choices worth calling out:

**Segments are defined by rule, not by KMeans.** Every user is labeled directly (`repeat_purchaser`, `one_time_buyer`, `cart_abandoner`, `browser`, `cold_user`) based on their actual behavior. This is more robust than clustering-then-naming: it guarantees the segment vocabulary always matches the brief's own language, and avoids clustering instability (collapse, micro-clusters, ambiguous cluster names).

**The ML stack doesn't pick products.** The training data uses anonymized item IDs (`ITEM6`, `CATEGORY_1`) with no relationship to the frontend's real catalog. The ML segment drives tone, hero copy, and CTA selection; `CatalogService` selects real catalog products by matching the visitor's `recent_categories` (a real, catalog-native signal already sent by the frontend) with exponential recency weighting.

## Backend setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

You need two things next to `main.py`:
- `chameleon_api_assets/` — trained model artifacts (produced by `run_pipeline.py`, not committed)
- `data/catalog.json` — real product catalog

Then:
```bash
uvicorn main:app --reload --port 8003
```

Health check: `http://127.0.0.1:8003/api/v1/health` should return `{"status":"ok","catalog_loaded":true}`.

Interactive docs: `http://127.0.0.1:8003/docs`.

### Training

`run_pipeline.py` is the training entry point. It expects the raw GA4-style activity and transaction CSVs. On Colab:

```python
!python run_pipeline.py
```

Outputs a `chameleon_api_assets/` folder containing:
- `kmeans.joblib`, `kmeans_scaler.joblib`, `feature_scaler.joblib`, `context_encoder.joblib`
- `random_forest.joblib`
- `deep_model.pth`, `lstm_model.pth`
- `numeric_feature_cols.json`, `event2id.json`, `cluster_names.json`
- `cold_start_resolver.joblib`

Unzip that folder into `backend/chameleon_api_assets/` for the API to load it on boot.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Opens on Vite's default port (`http://localhost:5173`). Navigate to `/showcase` to see the persona demo.

The frontend's API base URL is set in `src/services/api.ts` and `src/pages/PersonaShowcase.tsx`. Both default to `http://127.0.0.1:8003`.

## API contract

**`GET /api/v1/health`**
```json
{"status": "ok", "catalog_loaded": true}
```

**`POST /api/v1/personalize`**

Request:
```json
{
  "user_pseudo_id": "string",
  "total_events": 1,
  "unique_pages": 1,
  "total_revenue": 0,
  "days_since_first_visit": 0,
  "recent_sequence": ["view_item", "add_to_cart"],
  "recent_categories": ["Shoes"],
  "recent_display_categories": ["Shoes"]
}
```

Response:
```json
{
  "hero_section": {
    "title": "Explore Our Shoes Collection",
    "subtitle": "...",
    "cta": "Shop Now",
    "targeted_products": ["SH-B22E4A", "SH-CB7994", "..."]
  },
  "product_modules": [
    {"title": "Top Picks in Shoes", "products": ["SH-...", "..."]}
  ],
  "cta_modules": [
    {"type": "engagement", "title": "...", "button_text": "..."}
  ],
  "personalization_details": {
    "predicted_segment": 1,
    "segment_name": "one_time_buyer",
    "assigned_business_tags": ["standard"],
    "prediction_source": "deep_model",
    "confidence": 0.9995,
    "is_cold_start": false,
    "target_category": "Shoes"
  }
}
```

`prediction_source` is one of: `deep_model`, `random_forest_fallback`, `lstm_override`, `cold_start_lookup:<levels>`, `cold_start_global_default`.

## Documentation

`docs/Chameleon_Model_Documentation.docx` contains the full model writeup: architecture, engineering decisions, and results.

## License

Educational / hackathon submission.