import React from 'react';

import { ChameleonMascot } from './chameleonMascot';
import { usePersonalization } from '../context/personalizationContext';

export function Sidebar() {
  const { data, accent } = usePersonalization();
  const details = data?.personalization_details;

  return (
    <aside className="w-[260px] flex-shrink-0 flex flex-col h-screen sticky top-0" style={{ backgroundColor: '#171717' }}>
      {/* Logo */}
      <div className="h-20 flex-shrink-0 flex items-center px-4 -ml-2">
        <ChameleonMascot color={accent} size={45} />
        <h1 className="font-['Helvetica_Neue'] text-4xl text-white font-semibold whitespace-nowrap">Chameleon</h1>
      </div>

      {/* ML Telemetry -- vertically centered in the sidebar, dark-adapted, real data */}
      <div className="flex-1 flex items-center px-6">
        {details ? (
          <div
            className="w-full rounded-2xl p-4 overflow-hidden"
            style={{ backgroundColor: 'rgba(255,255,255,0.06)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.1)' }}
          >
            <div className="flex items-center justify-between mb-3 pb-2 border-b" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
              <div className="flex items-center gap-2">
                <div className="size-2 rounded-full animate-pulse" style={{ backgroundColor: accent }} />
                <span className="font-sans text-[11px] font-bold text-white">ML_TELEMETRY</span>
              </div>
              <span className="font-sans text-[10px]" style={{ color: '#6B6B6B' }}>LIVE</span>
            </div>
            <div className="flex flex-col gap-2 font-sans text-[11px] leading-relaxed group">
              <TelemetryRow label="segment" value={details.segment_name ?? String(details.predicted_segment)} accent={accent} dark />
              <TelemetryRow label="confidence" value={String(details.confidence)} accent={accent} highlight dark />
              <TelemetryRow label="source" value={details.prediction_source ?? '—'} accent={accent} dark />
              <div className="mt-2 border-t pt-2 flex flex-col gap-2" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                <TelemetryRow label="target_category" value={details.target_category ?? '—'} accent={accent} dark />
                <TelemetryRow label="is_cold_start" value={String(details.is_cold_start)} accent={accent} dark />
                <TelemetryRow label="business_tags" value={details.assigned_business_tags.join(', ')} accent={accent} dark />
              </div>
            </div>
          </div>
        ) : (
          <div
            className="w-full rounded-2xl p-4 overflow-hidden"
            style={{ backgroundColor: 'rgba(255,255,255,0.06)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.1)' }}
          >
            <div className="flex items-center justify-between mb-3 pb-2 border-b" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
              <div className="flex items-center gap-2">
                <div className="size-2 rounded-full" style={{ backgroundColor: '#6B6B6B' }} />
                <span className="font-sans text-[11px] font-bold text-white">ML_TELEMETRY</span>
              </div>
              <span className="font-sans text-[10px]" style={{ color: '#6B6B6B' }}>IDLE</span>
            </div>
            <p className="font-sans text-[11px] leading-relaxed" style={{ color: '#9CA3AF' }}>
              No telemetry yet — pick a visitor profile to activate the model.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}

function TelemetryRow({ label, value, accent, highlight, dark }: { label: string; value: string; accent: string; highlight?: boolean; dark?: boolean }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="flex-shrink-0" style={{ color: dark ? '#6B6B6B' : '#9CA3AF' }}>{label}:</span>
      <span className="font-bold text-right break-words min-w-0" style={{ color: highlight ? accent : dark ? '#FFFFFF' : '#121212' }}>{value}</span>
    </div>
  );
}