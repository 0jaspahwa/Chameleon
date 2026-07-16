import React from 'react';
import { Link } from 'react-router-dom';

import { getCatalogItem } from '../services/clickstream';
import { Sidebar } from '../components/sidebar';
import { PERSONAS, GLASS, GLASS_BORDER, usePersonalization } from '../context/personalizationContext';

/* ------------------------------------------------------------------ */

export function PersonaShowcase() {
  const { activeKey, data, loading, error, accent, runPersona } = usePersonalization();

  const details = data?.personalization_details;
  const heroProducts = (data?.hero_section.targeted_products ?? [])
    .map((id) => ({ id, details: getCatalogItem(id) }))
    .filter((p) => p.details)
    .slice(0, 4);

  return (
    <div className="min-h-screen w-full flex font-['Helvetica_Neue']" style={{ backgroundColor: '#F9F9F7', color: '#3E3E3E' }}>
      {/* ---------------- Sidebar ---------------- */}
      <Sidebar />

      {/* ---------------- Main ---------------- */}
      <div className="flex-1 relative overflow-hidden">
        {/* soft blurred blobs -- give the glass panels something to actually blur */}
        <div className="pointer-events-none fixed inset-0 overflow-hidden">
          <div
            className="absolute -top-24 left-1/4 w-[480px] h-[480px] rounded-full opacity-40 transition-[background] duration-700"
            style={{ background: `radial-gradient(circle, ${accent}55 0%, transparent 70%)`, filter: 'blur(90px)' }}
          />
          <div
            className="absolute top-1/3 -right-24 w-[420px] h-[420px] rounded-full opacity-30 transition-[background] duration-700"
            style={{ background: `radial-gradient(circle, ${accent}44 0%, transparent 70%)`, filter: 'blur(90px)' }}
          />
        </div>

        <div className="relative z-10">
          <header className="flex flex-col items-center gap-5 px-12 pt-8 pb-6 border-b flex-shrink-0" style={{ borderColor: 'rgba(18,18,18,0.08)' }}>
            <p className="font-['JetBrains_Mono'] text-[11px] tracking-[0.2em] uppercase" style={{ color: '#9CA3AF' }}>
              Live Personalization Engine
            </p>

            <div
              className="flex flex-wrap items-center gap-2 rounded-full p-1.5 w-fit"
              style={{ backgroundColor: GLASS, backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: GLASS_BORDER, boxShadow: '0 24px 48px -12px rgba(0,0,0,0.08)' }}
              role="group"
              aria-label="Choose a visitor profile"
            >
              {PERSONAS.map((persona) => {
                const isActive = persona.key === activeKey;
                return (
                  <button
                    key={persona.key}
                    onClick={() => runPersona(persona)}
                    aria-pressed={isActive}
                    className="font-['Helvetica_Neue'] text-sm font-medium px-4 py-2.5 rounded-full transition-all duration-300 ease-out hover:scale-110 hover:z-10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
                    style={{
                      backgroundColor: isActive ? '#FFFFFF' : 'transparent',
                      color: isActive ? '#121212' : '#9CA3AF',
                      boxShadow: isActive ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                      outlineColor: persona.accent,
                    }}
                  >
                    {persona.label}
                  </button>
                );
              })}
            </div>
          </header>

          {error && (
            <p className="font-['Helvetica_Neue'] text-[14px] text-center mt-6" style={{ color: '#E85D04' }}>
              {error}
            </p>
          )}

          {/* ---------------- Hero ---------------- */}
          <main className="px-6 md:px-12 pb-24 max-w-6xl mx-auto">
            {!activeKey ? (
              <div className="text-center py-24">
                <p className="font-['Helvetica_Neue'] text-[15px]" style={{ color: '#9CA3AF' }}>
                  Pick a visitor profile from the sidebar &mdash; the page is waiting to find out who's looking.
                </p>
              </div>
            ) : loading ? (
              <HeroSkeleton />
          ) : data ? (
            <>
              <div className="flex flex-col md:flex-row gap-12 items-center py-10">
                <div className="w-full md:w-1/2">
                  {heroProducts[0]?.details && (
                    <div className="relative rounded-3xl overflow-hidden shadow-[0_24px_48px_-12px_rgba(0,0,0,0.12)]">
                      <img
                        src={heroProducts[0].details.image_url}
                        alt={heroProducts[0].details.name}
                        className="w-full aspect-[4/5] object-cover"
                      />
                      {details && (
                        <div
                          className="absolute top-4 left-4 px-3 py-1.5 rounded-lg font-['JetBrains_Mono'] text-[11px] font-bold"
                          style={{ backgroundColor: GLASS, backdropFilter: 'blur(12px)', color: '#121212' }}
                        >
                          {Math.round((details.confidence ?? 0) * 100)}% Match
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <div className="w-full md:w-1/2 flex flex-col gap-6">
                  <div
                    className="inline-flex items-center gap-2 px-3 py-1 rounded-md w-fit font-['JetBrains_Mono'] text-[11px] font-bold uppercase tracking-wider transition-colors duration-700"
                    style={{ backgroundColor: `${accent}1A`, color: accent }}
                  >
                    {details?.target_category ?? 'Discovering'}
                  </div>
                  <h1 className="font-['Helvetica_Neue'] text-[44px] md:text-[56px] leading-[1.08]" style={{ color: '#121212' }}>
                    {data.hero_section.title}
                  </h1>
                  <p className="text-[17px] max-w-md" style={{ color: '#3E3E3E' }}>
                    {data.hero_section.subtitle}
                  </p>
                  <button
                    className="w-fit px-8 py-3.5 rounded-full text-white text-sm font-bold uppercase tracking-[0.1em] shadow-lg transition-colors duration-700"
                    style={{ backgroundColor: '#121212' }}
                  >
                    {data.hero_section.cta}
                  </button>
                </div>
              </div>

              {/* Product modules */}
              {data.product_modules.map((module, idx) => (
                <div key={idx} className="mt-16">
                  <h2 className="font-['Helvetica_Neue'] text-[28px] mb-6" style={{ color: '#121212' }}>{module.title}</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    {module.products.map((productId) => {
                      const p = getCatalogItem(productId);
                      if (!p) return null;
                      return (
                        <Link
                          key={productId}
                          to={`/product/${encodeURIComponent(productId)}`}
                          className="group focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 rounded-2xl"
                          style={{ outlineColor: accent }}
                        >
                          <div className="rounded-2xl overflow-hidden shadow-[0_12px_24px_-8px_rgba(0,0,0,0.1)]">
                            <img
                              src={p.image_url}
                              alt={p.name}
                              className="w-full aspect-[3/4] object-cover transition duration-500 group-hover:scale-105"
                            />
                          </div>
                          <p className="text-[14px] mt-3" style={{ color: '#121212' }}>{p.name}</p>
                          <p className="font-['JetBrains_Mono'] text-[13px] mt-0.5" style={{ color: '#9CA3AF' }}>${p.price}</p>
                        </Link>
                      );
                    })}
                  </div>
                </div>
              ))}
            </>
          ) : null}
          </main>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function HeroSkeleton() {
  return (
    <div className="flex flex-col md:flex-row gap-12 items-center py-10 animate-pulse">
      <div className="w-full md:w-1/2 aspect-[4/5] rounded-3xl" style={{ backgroundColor: 'rgba(18,18,18,0.06)' }} />
      <div className="w-full md:w-1/2 flex flex-col gap-4">
        <div className="h-6 w-32 rounded" style={{ backgroundColor: 'rgba(18,18,18,0.06)' }} />
        <div className="h-12 w-full rounded" style={{ backgroundColor: 'rgba(18,18,18,0.06)' }} />
        <div className="h-12 w-2/3 rounded" style={{ backgroundColor: 'rgba(18,18,18,0.06)' }} />
        <div className="h-4 w-full rounded" style={{ backgroundColor: 'rgba(18,18,18,0.06)' }} />
        <div className="h-11 w-40 rounded-full mt-2" style={{ backgroundColor: 'rgba(18,18,18,0.06)' }} />
      </div>
    </div>
  );
}