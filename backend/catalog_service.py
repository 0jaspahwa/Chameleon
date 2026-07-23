# -*- coding: utf-8 -*-
"""
catalog_service.py — bridges ML segment predictions to REAL, renderable
products.

WHY THIS EXISTS
----------------
The ML pipeline is trained on the anonymized hackathon dataset, whose
ItemCategory/ItemName values are placeholders ("CATEGORY_1", "ITEM6") with
no relationship to the frontend's real product catalog (catalog.json:
real IDs, real categories like "Shoes"/"Apparel"/"Furniture"). The model
can tell us WHO a visitor is (a segment/cluster + business tags), but it
can never tell us WHICH real product to show — that has to come from the
actual catalog, keyed by a real category name.

The frontend already gives us a real, catalog-native signal for free:
`recent_categories` / `recent_display_categories`, tracked client-side in
clickstream.ts from actual `catalog.json` items the visitor viewed. This
module uses that signal (not the ML model's category output) to select
real products.
"""

import json
import random
import logging
from collections import Counter
from typing import List, Optional, Dict

logger = logging.getLogger("catalog_service")


class CatalogService:
    def __init__(self, catalog_path: str):
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.by_category = {}
        for item in self.items:
            self.by_category.setdefault(item.get("category", "Unknown"), []).append(item)

        self.categories = list(self.by_category.keys())
        logger.info(f"CatalogService loaded {len(self.items)} items across {len(self.categories)} categories.")

    def resolve_target_category(self, recent_categories: List[str], recency_window: int = 8) -> str:
        """
        Picks a real catalog category to personalize around, weighted
        toward RECENT interest. A flat mode/count over history means a
        single fresh click can never beat an older category the visitor
        happened to browse more times in a past session -- personalization
        would only "catch up" once new clicks outnumber old ones, which is
        exactly the "have to explore a category more times" problem.

        Instead, each of the last `recency_window` views gets an
        exponentially increasing weight by recency (weight = 2^position,
        most recent = highest). Since 2^(n-1) > 2^(n-2)+...+2^0, the single
        MOST RECENT click always outweighs every older click combined --
        so one fresh click on a new category shows relevant products
        immediately, while a near-tie between two very recent clicks still
        blends sensibly instead of just hard-switching on the literal last
        pixel clicked.
        """
        cleaned = [c for c in (recent_categories or []) if c and c in self.by_category]
        if cleaned:
            recent_slice = cleaned[-recency_window:]
            scores: Dict[str, float] = {}
            for position, category in enumerate(recent_slice):
                scores[category] = scores.get(category, 0.0) + (2 ** position)
            return max(scores, key=scores.get)
        return random.choice(self.categories) if self.categories else "Unknown"

    def get_products(self, category: str, n: int = 4, exclude_ids: Optional[List[str]] = None) -> List[str]:
        """Returns up to n real catalog product IDs from a category."""
        pool = self.by_category.get(category, [])
        if not pool:
            pool = self.items
        exclude_ids = set(exclude_ids or [])
        candidates = [item["id"] for item in pool if item["id"] not in exclude_ids]
        random.shuffle(candidates)
        return candidates[:n]

    def get_secondary_category(self, primary_category: str) -> str:
        """A different category from the primary one, for a second module."""
        others = [c for c in self.categories if c != primary_category]
        return random.choice(others) if others else primary_category