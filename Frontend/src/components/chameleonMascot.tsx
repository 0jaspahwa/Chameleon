import React, { useState } from 'react';

interface ChameleonMascotProps {
  color: string;
  size?: number;
  className?: string;
}

/**
 * Monoline chameleon logomark, redrawn to match the geometric reference:
 * spiral tail, arched back into a flat-topped head step, hollow ring eye,
 * zigzag legs on a ground line. Single continuous stroke, no fill.
 * Color tracks the active persona accent (a literal callback to the
 * product's own name); tongue flicks out on hover or keyboard focus.
 */
export function ChameleonMascot({ color, size = 40, className = '' }: ChameleonMascotProps) {
  const [active, setActive] = useState(false);

  return (
    <svg
      viewBox="-30 0 240 50"
      width={size}
      height={size * (80 / 200)}
      className={`cursor-pointer outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 rounded-full overflow-visible ${className}`}
      style={{ outlineColor: color }}
      onMouseEnter={() => setActive(true)}
      onMouseLeave={() => setActive(false)}
      onFocus={() => setActive(true)}
      onBlur={() => setActive(false)}
      tabIndex={0}
      role="img"
      aria-label="Chameleon mark. Hover or focus to see its tongue flick out."
    >
      <g
        fill="none"
        stroke={color}
        strokeWidth={6}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-[stroke] duration-700"
      >
        {/* ground line */}
        <path d="M6,86 L194,86" />
        {/* spiral tail */}
        <path d="M62,86 C62,74 52,68 42,72 C33,76 31,87 39,91 C45,94 51,90 49,84 C48,80 43,80 43,84" />
        {/* back arch + flat head step + snout hook */}
        <path d="M64,86 C66,58 90,40 116,40 L142,40 C142,40 142,58 142,64 C155,64 165,58 168,50 C168,60 160,72 148,74 C143,75 140,72 141,66" />
        {/* neck zigzag legs */}
        <path d="M90,86 L101,66 L112,86 L123,66 L134,86" />
      </g>

      {/* hollow eye */}
      <circle cx="151" cy="52" r="6" fill="none" stroke={color} strokeWidth={4} className="transition-[stroke] duration-700" />

      {/* tongue */}
      <g
        className="transition-transform duration-300 ease-[cubic-bezier(.34,1.56,.64,1)] motion-reduce:transition-none motion-reduce:duration-0"
        style={{ transformOrigin: '168px 50px', transform: active ? 'scaleX(1)' : 'scaleX(0)' }}
      >
        <path d="M168,50 C182,46 193,48 200,42" fill="none" stroke="#E85D04" strokeWidth={5} strokeLinecap="round" />
        <circle cx="200" cy="42" r="4" fill="#E85D04" />
      </g>
    </svg>
  );
}