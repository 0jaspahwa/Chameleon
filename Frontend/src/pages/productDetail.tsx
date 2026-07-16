import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

import {
  getCatalogDisplayCategory,
  getCatalogItem,
  trackProductClick,
} from '../services/clickstream';

import { Sidebar } from '../components/sidebar';
import { usePersonalization } from '../context/personalizationContext';

interface ProductDetailProps {
  productId: string;
}

export function ProductDetail({ productId }: ProductDetailProps) {
  const product = getCatalogItem(productId);
  const { data, accent } = usePersonalization();
  const details = data?.personalization_details;

  const hasTracked = useRef(false);

  useEffect(() => {
    if (!hasTracked.current) {
      trackProductClick(productId);
      hasTracked.current = true;
    }
  }, [productId]);

  // Product Not Found
  if (!product) {
    return (
      <div className="min-h-screen flex font-['Helvetica_Neue']" style={{ backgroundColor: '#F9F9F7', color: '#3E3E3E' }}>
        <Sidebar />

        <main className="flex-1 flex items-center justify-center px-margin">
          <div className="max-w-md text-center">
            <h1 className="font-['Helvetica_Neue'] text-2xl font-bold mb-4" style={{ color: '#121212' }}>
              Product not found
            </h1>
            <Link className="font-semibold text-sm" style={{ color: accent }} to="/">
              Back to homepage
            </Link>
          </div>
        </main>
      </div>
    );
  }

  const isTargeted = details?.target_category === getCatalogDisplayCategory(product);

  return (
    <div className="min-h-screen flex font-['Helvetica_Neue']" style={{ backgroundColor: '#F9F9F7', color: '#3E3E3E' }}>
      {/* ---------------- Sidebar (shared with Persona Showcase) ---------------- */}
      <Sidebar />

      {/* ---------------- Main ---------------- */}
      <div className="flex-1 relative overflow-hidden">
        {/* soft blurred blobs -- matches the showcase page's glass aesthetic */}
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

        <main className="relative z-10 px-6 md:px-12 py-10 max-w-6xl mx-auto">
          <Link
            to="/home"
            className="inline-flex items-center gap-1 font-['JetBrains_Mono'] text-[11px] tracking-[0.15em] uppercase mb-8"
            style={{ color: '#9CA3AF' }}
          >
            &larr; Back to personalized homepage
          </Link>

          <section className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            {/* Product Image */}
            <div className="relative rounded-3xl overflow-hidden shadow-[0_24px_48px_-12px_rgba(0,0,0,0.12)]">
              <img
                alt={product.name}
                className="w-full aspect-[4/5] object-cover"
                src={product.image_url}
              />
              {isTargeted && details && (
                <div
                  className="absolute top-4 left-4 px-3 py-1.5 rounded-lg font-['JetBrains_Mono'] text-[11px] font-bold"
                  style={{ backgroundColor: 'rgba(255,255,255,0.65)', backdropFilter: 'blur(12px)', color: '#121212' }}
                >
                  {Math.round((details.confidence ?? 0) * 100)}% Match
                </div>
              )}
            </div>

            {/* Content */}
            <div className="flex flex-col gap-6">
              {/* Category */}
              <div
                className="inline-flex items-center gap-2 px-3 py-1 rounded-md w-fit font-['JetBrains_Mono'] text-[11px] font-bold uppercase tracking-wider transition-colors duration-700"
                style={{ backgroundColor: `${accent}1A`, color: accent }}
              >
                {getCatalogDisplayCategory(product)}
              </div>

              {/* Title */}
              <h1 className="font-['Helvetica_Neue'] text-[44px] md:text-[56px] leading-[1.08]" style={{ color: '#121212' }}>
                {product.name}
              </h1>

              {/* Description */}
              <p className="text-[17px] max-w-md" style={{ color: '#3E3E3E' }}>
                This view was tracked as a dataset-style view_item event for {productId}.
              </p>

              {/* Price */}
              <div className="font-['JetBrains_Mono'] text-2xl font-bold" style={{ color: '#121212' }}>
                ${product.price}
              </div>

              {/* CTA */}
              <Link
                to="/home"
                className="w-fit px-8 py-3.5 rounded-full text-white text-sm font-bold uppercase tracking-[0.1em] shadow-lg transition-colors duration-700"
                style={{ backgroundColor: '#121212' }}
              >
                Return to personalized homepage
              </Link>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}