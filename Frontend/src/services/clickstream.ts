import catalog from '../data/catalog.json';

export type CatalogItem = {
  id: string;
  name: string;
  category: string;
  price: number;
  image_url: string;
  description?: string;
  subcategory?: string;
  brand?: string;
  display_category?: string;
  original_price?: number;
  discount?: number;
  rating?: number;
};

const storageKey = 'chameleon_recent_product_events';

const maxEvents = 20;


/* -------------------------------- */
/* FAST LOOKUP MAP */
/* -------------------------------- */

const catalogMap: Record<string, CatalogItem> =
  Object.fromEntries(

    (catalog as CatalogItem[]).map((item) => [

      item.id,
      item

    ])

  );


/* -------------------------------- */
/* PRODUCT EVENTS */
/* -------------------------------- */

export type ProductClickEvent = {

  event_name: 'view_item';

  item_name: string;

  category: string;

  display_category: string;

  page_path: string;

};


/* -------------------------------- */
/* GET PRODUCT */
/* -------------------------------- */

export function getCatalogItem(productRef: string) {

  return (

    catalogMap[productRef] ??

    Object.values(catalogMap).find(

      (item) => item.name === productRef

    ) ??

    null

  );

}


/* -------------------------------- */
/* GET PRODUCTS BY CATEGORY */
/* -------------------------------- */

export function getProductsByCategory(category: string) {

  return (catalog as CatalogItem[]).filter(

    (item) => item.category === category

  );

}


/* -------------------------------- */
/* RANDOM PRODUCTS */
/* -------------------------------- */

export function getRandomProductsByCategory(

  category: string,
  limit = 4

) {

  const filtered = getProductsByCategory(category);

  return filtered
    .sort(() => 0.5 - Math.random())
    .slice(0, limit);

}


/* -------------------------------- */
/* DISPLAY CATEGORY */
/* -------------------------------- */

export function getCatalogDisplayCategory(

  item: CatalogItem

) {

  return item.display_category || item.category;

}


/* -------------------------------- */
/* TRACK PRODUCT CLICK */
/* -------------------------------- */

export function trackProductClick(productRef: string) {

  const item = getCatalogItem(productRef);

  const existing = getRecentProductEvents();

  const nextEvent: ProductClickEvent = {

    event_name: 'view_item',

    item_name: productRef,

    category: item?.category ?? 'UNKNOWN_CATEGORY',

    display_category: item
      ? getCatalogDisplayCategory(item)
      : 'Unknown',

    page_path: `/products/${productRef}/`,

  };

  localStorage.setItem(

    storageKey,

    JSON.stringify(

      [...existing, nextEvent].slice(-maxEvents)

    ),

  );

}


/* -------------------------------- */
/* RECENT EVENTS */
/* -------------------------------- */

export function getRecentProductEvents():

  ProductClickEvent[] {

  try {

    const parsed = JSON.parse(

      localStorage.getItem(storageKey) ?? '[]'

    );

    return Array.isArray(parsed)

      ? parsed.filter(

          (item): item is ProductClickEvent =>

            item?.event_name === 'view_item' &&
            typeof item.item_name === 'string',

        )

      : [];

  } catch {

    return [];

  }

}


/* -------------------------------- */
/* HELPERS */
/* -------------------------------- */

export function getRecentEventNames() {

  return getRecentProductEvents().map(

    (event) => event.event_name

  );

}


export function getRecentItemNames() {

  return getRecentProductEvents().map(

    (event) => event.item_name

  );

}


export function getRecentCategories() {

  return getRecentProductEvents().map(

    (event) => event.category

  );

}


export function getRecentDisplayCategories() {

  return getRecentProductEvents().map((event) => {

    if (event.display_category) {

      return event.display_category;

    }

    const item = getCatalogItem(event.item_name);

    return item
      ? getCatalogDisplayCategory(item)
      : 'Unknown';

  });

}