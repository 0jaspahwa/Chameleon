"""
generate_catalog_skeleton.py

Scans a folder of downloaded product images and generates a starter
catalog.json with one entry per image. Fill in the placeholder fields
(name, category, price, etc.) by hand -- that's the whole point of
building your own small catalog instead of fighting broken third-party
URLs.

USAGE
-----
    python generate_catalog_skeleton.py path/to/images/folder

This writes catalog.json in the current directory. Move it into
Frontend/src/data/catalog.json (or wherever CHAMELEON_CATALOG_PATH points
for the backend) when you're done editing it.

IMAGE PLACEMENT
----------------
Copy your image files into Frontend/public/images/ so Vite serves them
at /images/<filename> with zero config. The generated image_url values
below already assume that path.

CATEGORY NOTE
-------------
Your demo personas (personalizationContext.tsx) are tuned around these
category names: Apparel, Shoes, Accessories, Furniture, Office Gear.
Using the same names for `category` below means the persona buttons in
your demo will actually surface relevant products.
"""

import os
import sys
import json
import re

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def slugify_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").upper()
    return slug or "ITEM"


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_catalog_skeleton.py path/to/images/folder")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}")
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )

    if not files:
        print("No image files found in that folder.")
        sys.exit(1)

    catalog = []
    used_ids = set()

    for f in files:
        base_id = slugify_id(f)
        item_id = base_id
        suffix = 2
        while item_id in used_ids:
            item_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(item_id)

        catalog.append({
            "id": item_id,
            "name": "TODO: product name",
            "brand": "TODO: brand",
            "category": "TODO: Apparel | Shoes | Accessories | Furniture | Office Gear",
            "display_category": "TODO: subcategory (e.g. Bottomwear, Sneakers)",
            "price": 0.0,
            "original_price": 0.0,
            "discount": 0.0,
            "rating": "4.0",
            "image_url": f"/images/{f}",
            "description": "TODO: short description"
        })

    with open("catalog.json", "w", encoding="utf-8") as out:
        json.dump(catalog, out, indent=2, ensure_ascii=False)

    print(f"Wrote catalog.json with {len(catalog)} entries.")
    print("Next steps:")
    print("  1. Copy your image files into Frontend/public/images/")
    print("  2. Open catalog.json and fill in every TODO field")
    print("  3. Move catalog.json into Frontend/src/data/catalog.json")
    print("     (and update CHAMELEON_CATALOG_PATH for the backend if it points elsewhere)")


if __name__ == "__main__":
    main()