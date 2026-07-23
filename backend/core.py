# -*- coding: utf-8 -*-
"""
core.py — UltraPersonalizationEngine (Chameleon storefront)

WHAT CHANGED IN THIS PASS AND WHY
------------------------------------------------
1. STATELESS INFERENCE. The FastAPI deployment (main.py) has no live
   database of past user behavior — every request is served by a fresh
   process with no per-user history to look up. The old `generate_
   personalized_landing_page` assumed it could find the caller in
   `self.user_features` (a table only ever populated at training time),
   so in production that branch could never fire: every real request
   silently fell through to the cold-start path regardless of how much
   history the visitor actually had. Personalization now runs entirely
   off what's IN the request (total_events, recent_sequence, recent
   categories, ...), which is what the frontend actually sends. A
   database-backed lookup is still attempted first as a bonus if
   `user_features` happens to be populated (e.g. if you later wire up a
   real datastore), but it's no longer required for personalization to
   work.

2. REAL CATALOG-BACKED PRODUCT SELECTION. The ML model's ItemCategory/
   ItemName vocabulary comes from the anonymized training dataset
   ("CATEGORY_1", "ITEM6") and has no relationship to the frontend's
   actual product catalog (real IDs, real categories). The model can
   tell us WHO a visitor is (a behavioral segment + business tags) but
   never WHICH real product to show. Product selection now goes through
   `catalog_service.CatalogService`, keyed by the visitor's own recent
   category views (a real, catalog-native signal already sent by the
   frontend) — the ML segment instead drives tone/copy/CTA selection.

3. `hero_section` now includes `targeted_products` (a list of real
   catalog IDs), matching what Hero.tsx actually expects — previously
   missing entirely.

4. Added `UltraPersonalizationEngine.load_for_inference(...)`, a
   lightweight constructor that loads pre-trained assets directly without
   re-running the full training pipeline (which needs real historical
   data main.py doesn't have at boot). Replaces the previous approach of
   constructing a full engine against an empty dummy DataFrame and then
   monkey-patching internals after the fact.
"""

import json
import logging
import pandas as pd
import numpy as np

from ml_models import (
    PersonalizationModel,
    ColdStartResolver,
    MIN_EVENTS_FOR_BEHAVIORAL,
)
from catalog_service import CatalogService

logger = logging.getLogger("core")

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    print("Note: 'google-genai' not installed — AI-generated hero copy will use the static fallback.")
    HAS_GENAI = False


NUMERIC_FEATURE_COLS = ["total_events", "unique_pages", "total_revenue", "days_since_first_visit"]
CONTEXT_KEYS = ("category", "region", "country", "source", "medium", "Age")


class UltraPersonalizationEngine:
    def __init__(self, merged_df: pd.DataFrame, api_key: str = None, catalog_path: str = None):
        print("Initializing UltraPersonalizationEngine (training mode)...")
        self.merged_df = merged_df
        self.api_key = api_key
        self.catalog = CatalogService(catalog_path) if catalog_path else None

        self.pm = PersonalizationModel(rf_threshold=0.65, lstm_threshold=0.90)
        self.cold_start = ColdStartResolver()

        self.user_features = pd.DataFrame()

        self._prepare_user_features()
        self._generate_ground_truth_labels()
        self._apply_business_tags()
        self._fit_cold_start_resolver()
        self._train_behavioral_models()

        print("Engine ready: behavioral stack (RF/Deep/LSTM) + cold-start resolver both trained.")

    @classmethod
    def load_for_inference(cls, asset_dir: str, catalog_path: str, api_key: str = None):
        """
        Lightweight constructor for the live API: loads pre-trained assets
        directly, skipping the training pipeline entirely (there's no live
        historical DataFrame to train from at boot time). Personalization
        at inference time is stateless (see generate_personalized_landing_
        page), so `user_features`/`merged_df` are just kept as harmless
        empty placeholders rather than special-cased throughout.
        """
        import os
        import joblib
        import torch
        from ml_models import DeepPredictor, UserSequenceLSTM

        engine = cls.__new__(cls)  # bypass __init__ (no training data to train from)
        engine.merged_df = pd.DataFrame()
        engine.user_features = pd.DataFrame()
        engine.api_key = api_key
        engine.catalog = CatalogService(catalog_path)

        pm = PersonalizationModel(rf_threshold=0.65, lstm_threshold=0.90)
        pm.kmeans = joblib.load(os.path.join(asset_dir, "kmeans.joblib"))
        pm.kmeans_scaler = joblib.load(os.path.join(asset_dir, "kmeans_scaler.joblib"))
        pm.feature_scaler = joblib.load(os.path.join(asset_dir, "feature_scaler.joblib"))
        pm.context_encoder = joblib.load(os.path.join(asset_dir, "context_encoder.joblib"))

        with open(os.path.join(asset_dir, "numeric_feature_cols.json")) as f:
            pm.numeric_feature_cols = json.load(f)
        with open(os.path.join(asset_dir, "event2id.json")) as f:
            pm.event2id = json.load(f)

        rf_path = os.path.join(asset_dir, "random_forest.joblib")
        if os.path.exists(rf_path):
            pm.rf = joblib.load(rf_path)

        # IMPORTANT: pm.kmeans is an *internal auxiliary* clustering used only
        # as an extra engineered feature for RF/Deep -- its n_clusters has no
        # relationship to how many segment classes the classifier stack was
        # actually trained on. Read the true class count directly from the
        # saved checkpoint's output layer instead of guessing.
        deep_checkpoint = torch.load(os.path.join(asset_dir, "deep_model.pth"), map_location="cpu", weights_only=True)
        num_classes = deep_checkpoint["out.weight"].shape[0]

        context_dim = sum(len(cats) for cats in pm.context_encoder.categories_)
        deep_input_size = len(pm.numeric_feature_cols) + context_dim + 1  # +1 for appended cluster id
        pm.deep = DeepPredictor(input_size=deep_input_size, num_classes=num_classes)
        pm.deep.load_state_dict(deep_checkpoint)
        pm.deep.eval()

        lstm_path = os.path.join(asset_dir, "lstm_model.pth")
        if os.path.exists(lstm_path):
            pm.sequence_model = UserSequenceLSTM(vocab_size=len(pm.event2id), num_classes=num_classes, pad_idx=0)
            pm.sequence_model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
            pm.sequence_model.eval()

        engine.pm = pm

        cold_start_path = os.path.join(asset_dir, "cold_start_resolver.joblib")
        engine.cold_start = joblib.load(cold_start_path) if os.path.exists(cold_start_path) else ColdStartResolver()

        cluster_names_path = os.path.join(asset_dir, "cluster_names.json")
        if os.path.exists(cluster_names_path):
            with open(cluster_names_path) as f:
                # JSON keys are always strings; convert back to int cluster ids
                engine.cluster_names = {int(k): v for k, v in json.load(f).items()}
        else:
            engine.cluster_names = {i: name for i, name in enumerate(cls.PERSONA_ORDER)}

        print("UltraPersonalizationEngine loaded for inference (no training run).")
        return engine

    # ------------------------------------------------------------------
    # Training-time pipeline (used by run_pipeline.py, not by the live API)
    # ------------------------------------------------------------------
    def _prepare_user_features(self):
        print("   Step 1: Preparing per-user features (numeric + context)...")

        agg = {
            "total_events": ("event_name", "count"),
            "unique_pages": ("page_type", "nunique"),
            "total_revenue": ("purchase_revenue", "sum"),
            "num_purchases": ("event_name", lambda x: (x == "purchase").sum()),
            "num_cart_adds": ("event_name", lambda x: (x == "add_to_cart").sum()),
            "first_visit": ("eventTimestamp" if "eventTimestamp" in self.merged_df.columns else "timestamp", "min"),
        }
        for ctx_col in CONTEXT_KEYS:
            if ctx_col in self.merged_df.columns:
                agg[ctx_col] = (ctx_col, lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")

        features = self.merged_df.groupby("user_pseudo_id").agg(**agg).reset_index()

        features["first_visit"] = pd.to_datetime(features["first_visit"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        features["days_since_first_visit"] = (now - features["first_visit"]).dt.days.fillna(365)
        features = features.drop(columns=["first_visit"])

        fill_numeric = {c: 0 for c in NUMERIC_FEATURE_COLS if c in features.columns}
        features = features.fillna(value=fill_numeric)
        for ctx_col in CONTEXT_KEYS:
            if ctx_col in features.columns:
                features[ctx_col] = features[ctx_col].astype(str).fillna("unknown")

        self.user_features = features

    PERSONA_ORDER = ["repeat_purchaser", "one_time_buyer", "cart_abandoner", "browser", "cold_user"]

    def _generate_ground_truth_labels(self):
        print("   Step 2: Generating ground-truth behavioral segments (rule-based personas)...")
        self._assign_personas()

        self.user_features["is_high_intent"] = (
            (self.user_features["total_revenue"] > 0) | (self.user_features["total_events"] > 5)
        ).astype(int)

    def _assign_personas(self):
        """
        Assigns each user DIRECTLY to one of five business personas via
        rule, rather than running KMeans and naming clusters after the
        fact. This is more robust for two reasons: (1) it guarantees the
        segment vocabulary always matches the brief's own language exactly
        ("cart abandoners, frequent viewers, repeat purchasers"), and (2) it
        avoids KMeans-driven instability -- collapsed clusters, 3-6 user
        micro-clusters, or two different clusters both needing the same
        post-hoc name -- since every user gets an unambiguous, individually
        well-defined label regardless of how "clustered" the population
        actually is. Checked in order of specificity:
          repeat_purchaser -> more than one purchase
          one_time_buyer   -> exactly one purchase
          cart_abandoner   -> added to cart, never purchased
          browser          -> meaningful browsing activity, no cart/purchase
          cold_user        -> minimal activity of any kind
        """
        median_events = self.user_features["total_events"].median()

        def assign(row):
            if row["num_purchases"] > 1:
                return "repeat_purchaser"
            if row["num_purchases"] == 1:
                return "one_time_buyer"
            if row["num_cart_adds"] > 0:
                return "cart_abandoner"
            if row["total_events"] >= median_events:
                return "browser"
            return "cold_user"

        persona = self.user_features.apply(assign, axis=1)
        # Densify: map ONLY the personas that actually appear to contiguous
        # ids 0..k-1 (preserving PERSONA_ORDER). The old fixed 0..4 mapping
        # produced non-contiguous labels whenever a persona was absent, which
        # (a) crashed CrossEntropyLoss ("Target N is out of bounds", since
        # num_classes = nunique()), and (b) made argmax-index != class label
        # in the RF fallback / cluster_names lookup at inference.
        present_personas = [p for p in self.PERSONA_ORDER if (persona == p).any()]
        name_to_id = {name: i for i, name in enumerate(present_personas)}
        self.user_features["target_cluster"] = persona.map(name_to_id)
        self.cluster_names = {i: name for name, i in name_to_id.items()}
        print(f"   Persona distribution: {persona.value_counts().to_dict()}")

    def _apply_business_tags(self):
        print("   Step 3: Applying rule-based business tags...")

        def get_tags(row):
            tags = []
            if row["total_revenue"] > 200:
                tags.append("high_value")
            if row["days_since_first_visit"] < 30:
                tags.append("new_user")
            if row["total_events"] > 50:
                tags.append("frequent_shopper")
            return tags if tags else ["standard"]

        self.user_features["tags"] = self.user_features.apply(get_tags, axis=1)

    def _fit_cold_start_resolver(self):
        print("   Step 4: Fitting cold-start (demographic/context) fallback resolver...")
        self.cold_start.fit(self.user_features, self.merged_df)

    def _train_behavioral_models(self):
        print("   Step 5: Training behavioral stack on the full user population...")
        # Rule-based persona labels are well-defined for every user
        # regardless of activity level (even a 1-event user has an
        # unambiguous correct label, e.g. cold_user), so -- unlike the old
        # KMeans-based approach -- there's no reason to exclude low-activity
        # users from training. This also keeps cold_user/browser properly
        # represented instead of thinning them out. MIN_EVENTS_FOR_BEHAVIORAL
        # is still used separately at inference time (see _resolve_segment)
        # to decide whether a given LIVE request has enough signal to trust
        # the trained stack vs. the cold-start resolver -- that's a
        # different question from how the models were trained.
        all_users = self.user_features.copy()

        if len(all_users) < 30 or all_users["target_cluster"].nunique() < 2:
            print("   WARNING: not enough data to train a reliable behavioral stack. "
                  "The engine will rely on the cold-start resolver for all requests until more data is available.")
            return

        self.pm.train_all(
            df=all_users,
            target_col="target_cluster",
            merged_df=self.merged_df,
            numeric_feature_cols=NUMERIC_FEATURE_COLS,
        )
        self._print_accuracy_report()

    def _print_accuracy_report(self):
        m = getattr(self.pm, "eval_metrics", {}) or {}
        if not m:
            print("   No evaluation metrics available (behavioral stack was not trained).")
            return

        print("\n" + "=" * 60)
        print("ACCURACY REPORT (held-out validation set)")
        print("=" * 60)
        baseline = m.get("majority_baseline_accuracy")
        if baseline is not None:
            print(f"Majority-class baseline accuracy: {baseline:.4f}  "
                  f"(what you'd get by always guessing the most common segment)\n")

        for name, key in (("Random Forest", "random_forest"), ("Deep Predictor", "deep_model"), ("LSTM", "lstm")):
            if key in m:
                print(f"{name:<16} accuracy: {m[key]['accuracy']:.4f}   f1 (weighted): {m[key]['f1_weighted']:.4f}")
            else:
                print(f"{name:<16} not evaluated (model not trained / insufficient data)")

        if "deployed_pipeline" in m:
            dp = m["deployed_pipeline"]
            print(f"\nDEPLOYED PIPELINE (Deep -> RF fallback -> LSTM override, "
                  f"exactly as generate_personalized_landing_page runs it)")
            print(f"n={dp['n_evaluated']} validation users   "
                  f"prediction source breakdown: {dp['prediction_source_breakdown']}\n")
            if dp.get("classification_report_text"):
                print(dp["classification_report_text"])
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Inference (used by both run_pipeline.py's sanity tests and main.py)
    # ------------------------------------------------------------------
    def generate_personalized_landing_page(self, user_profile: dict) -> dict:
        """
        Stateless: everything needed is read directly from user_profile.

        Expected keys (all optional, sensible defaults applied):
          - total_events, unique_pages, total_revenue, days_since_first_visit:
            numeric signals. The frontend derives these from the current
            session (e.g. total_events = number of tracked view events),
            which is exactly what a stateless API should use.
          - recent_sequence: list of event names for this session (feeds
            the LSTM if there's enough of it).
          - recent_categories / recent_display_categories: REAL catalog
            category names the visitor actually viewed this session --
            drives which real products get recommended.
          - category, region, country, source, medium, Age: demographic/
            device context, if available (defaults to "unknown" -- the
            cold-start resolver backs off gracefully when these are
            unknown, see ml_models.ColdStartResolver).
          - user_pseudo_id: optional. Only used if a live `user_features`
            table happens to be populated (bonus path, not required).
        """
        numeric_features = {
            "total_events": float(user_profile.get("total_events", 1)),
            "unique_pages": float(user_profile.get("unique_pages", 1)),
            "total_revenue": float(user_profile.get("total_revenue", 0.0)),
            "days_since_first_visit": float(user_profile.get("days_since_first_visit", 0)),
        }
        context = {c: str(user_profile.get(c, "unknown")) for c in CONTEXT_KEYS}
        recent_sequence = user_profile.get("recent_sequence") or None
        recent_categories = user_profile.get("recent_categories") or []
        recent_display_categories = user_profile.get("recent_display_categories") or recent_categories

        cluster, pred_source, confidence, user_tags, fallback_avg_revenue = self._resolve_segment(
            user_profile, numeric_features, context, recent_sequence
        )
        segment_profile = self._get_segment_profile(cluster)
        # In the deployed stateless path `user_features` is empty, so
        # _get_segment_profile returns zeros and the cold-start resolver's
        # computed avg_revenue never reached hero copy / CTA tiering. Fall
        # back to the best revenue signal we actually have (cold-start segment
        # average, or the visitor's own revenue) so it isn't stuck at $0.00.
        if not segment_profile.get("avg_revenue") and fallback_avg_revenue:
            segment_profile["avg_revenue"] = float(fallback_avg_revenue)

        target_category = self.catalog.resolve_target_category(recent_display_categories) if self.catalog else "Unknown"
        secondary_category = self.catalog.get_secondary_category(target_category) if self.catalog else target_category

        hero_section = self._generate_hero_section(target_category, segment_profile)
        product_modules = self._generate_product_modules(target_category, secondary_category)
        cta_modules = self._generate_cta_modules(segment_profile, user_tags)

        is_cold_start = pred_source.startswith("cold_start")
        segment_name = getattr(self, "cluster_names", {}).get(int(cluster), f"segment_{cluster}")
        return {
            "hero_section": hero_section,
            "product_modules": product_modules,
            "cta_modules": cta_modules,
            "personalization_details": {
                "predicted_segment": int(cluster),
                "segment_name": segment_name,
                "assigned_business_tags": user_tags,
                "prediction_source": pred_source,
                "confidence": round(confidence, 4),
                "is_cold_start": is_cold_start,
                "target_category": target_category,
            },
        }

    def _resolve_segment(self, user_profile: dict, numeric_features: dict, context: dict, recent_sequence):
        """
        Decides whether there's enough signal in THIS request to trust the
        behavioral stack, or whether to fall back to the cold-start
        resolver. Optionally tries a live DB lookup first as a bonus if
        `user_features` happens to be populated (harmless no-op otherwise).
        """
        user_id = user_profile.get("user_pseudo_id")
        if user_id is not None and not self.user_features.empty:
            match = self.user_features[self.user_features["user_pseudo_id"] == user_id]
            if not match.empty and match.iloc[0]["total_events"] >= MIN_EVENTS_FOR_BEHAVIORAL and self.pm.deep is not None:
                row = match.iloc[0]
                numeric_features = {c: float(row[c]) for c in NUMERIC_FEATURE_COLS}
                context = {c: str(row.get(c, "unknown")) for c in CONTEXT_KEYS}
                tags = self._business_tags_from_numeric(numeric_features)
                pred_info = self.pm.predict_for_user(numeric_features, context, user_sequence=recent_sequence)
                # DB path: user_features is populated, so _get_segment_profile
                # will supply a real segment average -> no fallback needed.
                return pred_info["cluster"], pred_info["source"], pred_info["confidence"], tags, None

        signal_strength = len(recent_sequence or []) or numeric_features.get("total_events", 0)
        tags = self._business_tags_from_numeric(numeric_features)

        if signal_strength >= MIN_EVENTS_FOR_BEHAVIORAL and self.pm.deep is not None:
            try:
                pred_info = self.pm.predict_for_user(numeric_features, context, user_sequence=recent_sequence)
                # Live behavioral path: no segment table to average over, so
                # use the visitor's own revenue as the best available proxy.
                return (pred_info["cluster"], pred_info["source"], pred_info["confidence"],
                        tags, numeric_features.get("total_revenue", 0.0))
            except Exception as e:
                logger.warning(f"Behavioral prediction failed, falling back to cold start: {e}")

        resolved = self.cold_start.resolve(context)
        if "new_user" not in tags:
            tags.append("new_user")
        if resolved.get("avg_revenue", 0) > 200 and "high_value_segment" not in tags:
            tags.append("high_value_segment")
        # Cold-start path: surface the resolved segment's avg_revenue so it can
        # drive hero copy / CTA tiering instead of being dropped on the floor.
        return resolved["cluster"], resolved["source"], 0.5, tags, resolved.get("avg_revenue", 0.0)

    # -- helpers --------------------------------------------------------
    def _business_tags_from_numeric(self, numeric_features: dict) -> list:
        tags = []
        if numeric_features.get("total_revenue", 0) > 200:
            tags.append("high_value")
        if numeric_features.get("days_since_first_visit", 365) < 30:
            tags.append("new_user")
        if numeric_features.get("total_events", 0) > 50:
            tags.append("frequent_shopper")
        return tags if tags else ["standard"]

    def _get_segment_profile(self, cluster: int) -> dict:
        if self.user_features.empty:
            return {"avg_revenue": 0.0, "avg_events": 0.0}
        cluster_data = self.user_features[self.user_features.get("target_cluster", -1) == cluster]
        if cluster_data.empty:
            return {"avg_revenue": 0.0, "avg_events": 0.0}
        return {
            "avg_revenue": float(cluster_data["total_revenue"].mean()),
            "avg_events": float(cluster_data["total_events"].mean()),
        }

    def _generate_hero_section(self, target_category: str, segment_profile: dict) -> dict:
        targeted_products = self.catalog.get_products(target_category, n=4) if self.catalog else []

        copy = None
        if HAS_GENAI and self.api_key:
            try:
                client = genai.Client(api_key=self.api_key)
                prompt = f"""
                As a marketing expert for an e-commerce store, write content for a website's hero section.
                The visitor's inferred favorite category is '{target_category}'.
                Their segment has an average purchase value of ${segment_profile.get('avg_revenue', 0):.2f}.
                Return JSON with exactly three keys: "title" (max 8 words), "subtitle" (one engaging sentence),
                "cta" (max 4 words). Avoid generic phrases.
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                copy = json.loads(response.text)
            except Exception as e:
                print(f"Generative AI hero-copy error, using static fallback: {e}")

        if copy is None:
            copy = {
                "title": f"Explore Our {target_category} Collection",
                "subtitle": "Find exactly what you need from our handpicked selection.",
                "cta": "Shop Now",
            }

        copy["targeted_products"] = targeted_products
        return copy

    def _generate_product_modules(self, target_category: str, secondary_category: str) -> list:
        if not self.catalog:
            return []

        modules = []
        primary_products = self.catalog.get_products(target_category, n=4)
        if primary_products:
            modules.append({"title": f"Top Picks in {target_category}", "products": primary_products})

        secondary_products = self.catalog.get_products(secondary_category, n=4, exclude_ids=primary_products)
        if secondary_products:
            modules.append({"title": f"Trending in {secondary_category}", "products": secondary_products})

        return modules

    def _generate_cta_modules(self, segment_profile: dict, user_tags: list) -> list:
        ctas = []
        if "high_value" in user_tags or "high_value_segment" in user_tags:
            ctas.append({"type": "loyalty", "title": "An Exclusive Offer for Our VIPs", "button_text": "View Your Deals"})
        elif "new_user" in user_tags:
            ctas.append({"type": "first_purchase", "title": "Get 20% Off Your First Order!", "button_text": "Claim Offer"})
        elif segment_profile.get("avg_revenue", 0) < 100:
            ctas.append({"type": "engagement", "title": "Explore Our Best-Sellers", "button_text": "See What's Popular"})
        else:
            ctas.append({"type": "standard", "title": "Discover Something New", "button_text": "Browse All"})
        return ctas