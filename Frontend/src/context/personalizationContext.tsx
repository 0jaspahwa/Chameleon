import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';

import type { PersonalizeResponse } from '../types';

/* ------------------------------------------------------------------ */
/* Persona demo profiles                                               */
/* ------------------------------------------------------------------ */

export type PersonaKey = 'cold_user' | 'browser' | 'cart_abandoner' | 'one_time_buyer' | 'repeat_purchaser';

export interface PersonaDef {
  key: PersonaKey;
  label: string;
  accent: string;
  payload: Record<string, unknown>;
}

export const PERSONAS: PersonaDef[] = [
  {
    key: 'cold_user',
    label: 'New Visitor',
    accent: '#6B7280',
    payload: {
      user_pseudo_id: 'DEMO-COLD-01',
      total_events: 1, unique_pages: 1, total_revenue: 0, days_since_first_visit: 0,
      recent_sequence: [], recent_items: [], recent_categories: [], recent_display_categories: [],
    },
  },
  {
    key: 'browser',
    label: 'Browser',
    accent: '#6B9B5E',
    payload: {
      user_pseudo_id: 'DEMO-BROWSER-01',
      total_events: 8, unique_pages: 6, total_revenue: 0, days_since_first_visit: 2,
      recent_sequence: ['page_view', 'page_view', 'view_item', 'page_view', 'view_item', 'page_view'],
      recent_items: [], recent_categories: ['Apparel', 'Apparel', 'Shoes'], recent_display_categories: ['Apparel', 'Apparel', 'Shoes'],
    },
  },
  {
    key: 'cart_abandoner',
    label: 'Cart Abandoner',
    accent: '#E85D04',
    payload: {
      user_pseudo_id: 'DEMO-CART-01',
      total_events: 14, unique_pages: 8, total_revenue: 0, days_since_first_visit: 5,
      recent_sequence: ['page_view', 'view_item', 'view_item', 'add_to_cart', 'page_view', 'view_item'],
      recent_items: [], recent_categories: ['Accessories', 'Accessories'], recent_display_categories: ['Accessories', 'Accessories'],
    },
  },
  {
    key: 'one_time_buyer',
    label: 'One-Time Buyer',
    accent: '#3E8E9E',
    payload: {
      user_pseudo_id: 'DEMO-ONETIME-01',
      total_events: 20, unique_pages: 10, total_revenue: 65, days_since_first_visit: 14,
      recent_sequence: ['view_item', 'add_to_cart', 'purchase', 'page_view', 'view_item'],
      recent_items: [], recent_categories: ['Furniture', 'Furniture'], recent_display_categories: ['Furniture', 'Furniture'],
    },
  },
  {
    key: 'repeat_purchaser',
    label: 'Repeat Purchaser',
    accent: '#C79A2E',
    payload: {
      user_pseudo_id: 'DEMO-REPEAT-01',
      total_events: 55, unique_pages: 22, total_revenue: 340, days_since_first_visit: 180,
      recent_sequence: ['view_item', 'purchase', 'view_item', 'purchase', 'view_item', 'add_to_cart', 'purchase'],
      recent_items: [], recent_categories: ['Office Gear', 'Office Gear'], recent_display_categories: ['Office Gear', 'Office Gear'],
    },
  },
];

export const API_BASE = 'http://127.0.0.1:8000';
export const GLASS = 'rgba(255,255,255,0.65)';
export const GLASS_BORDER = '1px solid rgba(255,255,255,0.6)';

const STORAGE_KEY = 'chameleon:lastPersonalization';

interface StoredState {
  activeKey: PersonaKey;
  data: PersonalizeResponse;
}

interface PersonalizationContextValue {
  activeKey: PersonaKey | null;
  data: PersonalizeResponse | null;
  loading: boolean;
  error: string | null;
  accent: string;
  active: PersonaDef | null;
  runPersona: (persona: PersonaDef) => Promise<void>;
}

const PersonalizationContext = createContext<PersonalizationContextValue | null>(null);

export function PersonalizationProvider({ children }: { children: ReactNode }) {
  const [activeKey, setActiveKey] = useState<PersonaKey | null>(null);
  const [data, setData] = useState<PersonalizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Hydrate from the last persona run so the sidebar's ML telemetry stays
  // populated when navigating between pages (Persona Showcase <-> Product Detail).
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as StoredState;
        setActiveKey(parsed.activeKey);
        setData(parsed.data);
      }
    } catch {
      // ignore malformed/missing cache
    }
  }, []);

  const active = PERSONAS.find((p) => p.key === activeKey) ?? null;
  const accent = active?.accent ?? '#9CA3AF';

  const runPersona = useCallback(async (persona: PersonaDef) => {
    setActiveKey(persona.key);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/personalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(persona.payload),
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const json = (await res.json()) as PersonalizeResponse;
      setData(json);
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ activeKey: persona.key, data: json }));
      } catch {
        // storage full/unavailable -- non-fatal, sidebar just won't persist across pages
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? `Couldn't reach the model: ${err.message}. Is the API running at ${API_BASE}?`
          : 'Something went wrong.'
      );
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <PersonalizationContext.Provider value={{ activeKey, data, loading, error, accent, active, runPersona }}>
      {children}
    </PersonalizationContext.Provider>
  );
}

export function usePersonalization() {
  const ctx = useContext(PersonalizationContext);
  if (!ctx) {
    throw new Error('usePersonalization must be used within a PersonalizationProvider');
  }
  return ctx;
}