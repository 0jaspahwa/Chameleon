import type { PersonalizeResponse } from '../types';
import { getRecentCategories, getRecentDisplayCategories, getRecentEventNames, getRecentItemNames } from './clickstream';

function normalizePersonalizationResponse(raw: any): PersonalizeResponse {
  const nested = raw?.hero_section;
  const nestedHero = nested?.hero_section;

  return {
    hero_section: nestedHero ?? nested ?? {
      title: 'Welcome to UltraPersonal.',
      subtitle: 'Discover curated products tailored to your behavior.',
      cta: 'Start Exploring',
    },
    product_modules: Array.isArray(raw?.product_modules)
      ? raw.product_modules
      : Array.isArray(nested?.product_modules)
        ? nested.product_modules
        : [],
    personalization_details:
      raw?.personalization_details ??
      nested?.personalization_details ?? {
        predicted_segment: 0,
        assigned_business_tags: ['fallback_mode'],
      },
  };
}

// In-memory cache for the current browser tab's session. Without this, every
// time a visitor goes to a product page and comes back to /home, React
// Router remounts the homepage and re-fires the fetch from scratch --
// showing the skeleton loader again even though they likely just saw nearly
// the same data seconds ago. `getCachedPersonalizationData` lets the
// homepage render instantly from the last known response, while
// `fetchPersonalizationData` still refreshes it in the background (e.g. to
// pick up a category shift from the product they just viewed).
let cachedResponse: PersonalizeResponse | null = null;

export function getCachedPersonalizationData(): PersonalizeResponse | null {
  return cachedResponse;
}

export async function fetchPersonalizationData(): Promise<PersonalizeResponse> {
  const recentSequence = getRecentEventNames();
  const recentItems = getRecentItemNames();
  const recentCategories = getRecentCategories();
  const recentDisplayCategories = getRecentDisplayCategories();

  // Hit your live local FastAPI server
  const response = await fetch('http://127.0.0.1:8000/api/v1/personalize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    // We send a cold-start profile. In a real app, you'd pull this from cookies/local storage.
    body: JSON.stringify({
      user_pseudo_id: "FRONTEND-TEST-USER-01",
      total_events: Math.max(1, recentSequence.length),
      unique_pages: Math.max(1, recentItems.length),
      total_revenue: 0.0,
      days_since_first_visit: 0,
      recent_sequence: recentSequence,
      recent_items: recentItems,
      recent_categories: recentCategories,
      recent_display_categories: recentDisplayCategories,
    })
  });

  if (!response.ok) {
    throw new Error('Failed to fetch from backend');
  }

  const normalized = normalizePersonalizationResponse(await response.json());
  cachedResponse = normalized;
  return normalized;
}