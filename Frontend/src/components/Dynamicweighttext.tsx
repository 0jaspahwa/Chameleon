// DynamicWeightText — adapted from the Originkit "Dynamic Weight" Framer
// component. Renders text whose letters individually morph their font
// `wght` (variable-font weight axis) based on proximity to the cursor.
//
// Requires `framer-motion` as a dependency (npm install framer-motion).

import * as React from "react"
import { useEffect, useRef } from "react"
import { motion, useAnimationFrame } from "framer-motion"

export type DynamicWeightTextProps = {
  label: string
  fromWeight?: number
  toWeight?: number
  /** 1–100, how far (in px, up to MAX_REACH) the cursor influence reaches */
  strength?: number
  fontSize?: number | string
  color?: string
  /** Easing speed of the weight ramp, in seconds */
  duration?: number
  className?: string
  style?: React.CSSProperties
}

const MAX_REACH = 800

// Bundled variable font so the wght morph works without extra setup.
const INTER_VARIABLE_FONT_FACE = `
@font-face {
    font-family: "InterVariableDW";
    src: url("https://rsms.me/inter/font-files/InterVariable.woff2?v=4.0") format("woff2-variations");
    font-weight: 100 900;
    font-style: normal;
    font-display: swap;
}
`

const VARIABLE_FONT_STACK =
  '"InterVariableDW", "Inter Variable", "Inter", system-ui, sans-serif'

export default function DynamicWeightText({
  label,
  fromWeight = 400,
  toWeight = 900,
  strength = 25,
  fontSize = 48,
  color = "#17151d",
  duration = 0.3,
  className,
  style,
}: DynamicWeightTextProps) {
  const reach = Math.max(1, (Math.max(1, Math.min(100, strength)) / 100) * MAX_REACH)

  const containerRef = useRef<HTMLDivElement>(null)
  const letterRefs = useRef<Array<HTMLSpanElement | null>>([])
  const letterFactorsRef = useRef<number[]>([])
  const lastFrameRef = useRef(0)
  const mousePositionRef = useRef({ x: -99999, y: -99999 })

  useEffect(() => {
    const updatePosition = (clientX: number, clientY: number) => {
      const el = containerRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      mousePositionRef.current = { x: clientX - rect.left, y: clientY - rect.top }
    }
    const handleMouseMove = (ev: MouseEvent) => updatePosition(ev.clientX, ev.clientY)
    const handleTouchMove = (ev: TouchEvent) => {
      if (ev.touches.length === 0) return
      updatePosition(ev.touches[0].clientX, ev.touches[0].clientY)
    }
    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("touchmove", handleTouchMove)
    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("touchmove", handleTouchMove)
    }
  }, [])

  const fromSettings = `'wght' ${fromWeight}`

  useAnimationFrame((now: number) => {
    const container = containerRef.current
    if (!container) return
    const containerRect = container.getBoundingClientRect()
    const mx = mousePositionRef.current.x
    const my = mousePositionRef.current.y

    const prevT = lastFrameRef.current || now
    const dtSec = Math.min(0.1, Math.max(0, (now - prevT) / 1000))
    lastFrameRef.current = now

    const tau = Math.max(0.016, duration)
    const a = 1 - Math.exp(-dtSec / tau)

    for (let i = 0; i < letterRefs.current.length; i++) {
      const letterEl = letterRefs.current[i]
      if (!letterEl) continue
      const rect = letterEl.getBoundingClientRect()
      const cx = rect.left + rect.width / 2 - containerRect.left
      const cy = rect.top + rect.height / 2 - containerRect.top
      const dx = mx - cx
      const dy = my - cy
      const dist = Math.sqrt(dx * dx + dy * dy)

      const target = Math.min(Math.max(1 - dist / reach, 0), 1)
      const prev = letterFactorsRef.current[i] ?? 0
      const f = prev + (target - prev) * a
      letterFactorsRef.current[i] = f

      if (f < 0.001) {
        if (letterEl.style.fontVariationSettings !== fromSettings) {
          letterEl.style.fontVariationSettings = fromSettings
        }
        continue
      }

      const w = Math.round(fromWeight + (toWeight - fromWeight) * f)
      letterEl.style.fontVariationSettings = `'wght' ${w}`
    }
  })

  const innerSpanStyle: React.CSSProperties = {
    fontFamily: VARIABLE_FONT_STACK,
    fontSize,
    color,
    display: "inline-block",
  }

  const words = label ? label.split(" ") : []
  letterRefs.current = []
  let letterIndex = 0

  return (
    <span ref={containerRef} className={className} style={{ display: "inline-block", ...style }}>
      <style>{INTER_VARIABLE_FONT_FACE}</style>
      {words.length === 0 ? null : (
        <span style={innerSpanStyle}>
          <span
            style={{
              position: "absolute",
              width: 1,
              height: 1,
              padding: 0,
              margin: -1,
              overflow: "hidden",
              clip: "rect(0,0,0,0)",
              whiteSpace: "nowrap",
              borderWidth: 0,
            }}
          >
            {label}
          </span>
          {words.map((word, wi) => {
            const wordLetters = word.split("")
            return (
              <React.Fragment key={wi}>
                <span aria-hidden style={{ display: "inline-block", whiteSpace: "nowrap" }}>
                  {wordLetters.map((letter, li) => {
                    const idx = letterIndex++
                    return (
                      <motion.span
                        key={li}
                        ref={(el: HTMLSpanElement | null) => {
                          letterRefs.current[idx] = el
                        }}
                        style={{ display: "inline-block", fontVariationSettings: fromSettings }}
                      >
                        {letter}
                      </motion.span>
                    )
                  })}
                </span>
                {wi < words.length - 1 && (
                  <span aria-hidden style={{ display: "inline-block" }}>
                    &nbsp;
                  </span>
                )}
              </React.Fragment>
            )
          })}
        </span>
      )}
    </span>
  )
}