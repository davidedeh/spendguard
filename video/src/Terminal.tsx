import { Fragment } from 'react'
import { interpolate, useCurrentFrame } from 'remotion'

import { segments } from './ansi'
import type { CapturedLine } from './data'
import { fps } from './format'
import { between, easeInOut, entrance } from './motion'
import { contentDelay } from './pacing'
import { colours, fonts, type Tone } from './theme'
import { WindowTitleBar } from './WindowTitleBar'

export type Highlight = { line: number; from: number; to?: number; tone: Tone; callout?: string }

export type WorkerChip = {
  beforeLine: number
  at: number
  duration: number
  realSeconds: number
  label: string
}

const lineHeight = 1.5
const textWidth = 1260
const gutterWidth = 340

const toneColour = (tone: Tone) => colours[tone]

const lineVisibility = (highlights: Highlight[], frame: number, line: number) => {
  const active = highlights.filter(highlight => between(frame, highlight.from, highlight.to ?? Number.POSITIVE_INFINITY) > 0)
  if (active.length === 0) {
    return 1
  }
  const focus = Math.max(
    ...active.map(highlight => between(frame, highlight.from, highlight.to ?? Number.POSITIVE_INFINITY)),
  )
  return active.some(highlight => highlight.line === line) ? 1 : 1 - 0.6 * focus
}

const Chip = ({ chip, frame }: { chip: WorkerChip; frame: number }) => {
  const shown = entrance(frame, chip.at, 10)
  const progress = interpolate(frame, [chip.at, chip.at + chip.duration], [0, 1], {
    easing: easeInOut,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  })
  const speedUp = Math.round(chip.realSeconds / (chip.duration / fps))
  const done = progress >= 1
  return (
    <div
      style={{
        opacity: shown,
        margin: '10px 0 12px 2ch',
        display: 'flex',
        alignItems: 'center',
        gap: 18,
        fontFamily: fonts.sans,
        fontSize: 24,
        color: colours.dim,
      }}
    >
      <div style={{ width: 220, height: 8, borderRadius: 4, background: colours.panelBorder, overflow: 'hidden' }}>
        <div style={{ width: `${progress * 100}%`, height: '100%', background: done ? colours.green : colours.accent }} />
      </div>
      <span style={{ fontVariantNumeric: 'tabular-nums', color: colours.text, fontWeight: 600, minWidth: 80 }}>
        {(progress * chip.realSeconds).toFixed(1)}s
      </span>
      <span>
        {chip.label} · {done ? 'real time' : `shown ${speedUp}× faster`}
      </span>
    </div>
  )
}

export const Terminal = ({
  title,
  lines,
  appearAt,
  highlights = [],
  chip,
  hidden = [],
  fontSize = 28,
  footnote,
}: {
  title: string
  lines: CapturedLine[]
  appearAt: number[]
  highlights?: Highlight[]
  chip?: WorkerChip
  hidden?: number[]
  fontSize?: number
  footnote: string
}) => {
  const frame = useCurrentFrame()
  const windowShown = entrance(frame, contentDelay, 18)
  return (
    <div
      style={{
        position: 'absolute',
        left: 120,
        top: 250,
        width: textWidth + gutterWidth + 100,
        opacity: windowShown,
        transform: `translateY(${(1 - windowShown) * 24}px)`,
      }}
    >
      <div
        style={{
          background: colours.panel,
          border: `1px solid ${colours.panelBorder}`,
          borderRadius: 18,
          boxShadow: '0 30px 80px rgba(0, 0, 0, 0.45)',
          overflow: 'hidden',
        }}
      >
        <WindowTitleBar
          title={title}
          style={{ padding: '16px 22px', borderBottom: `1px solid ${colours.panelBorder}` }}
        />
        <div style={{ padding: '26px 36px 30px', fontFamily: fonts.mono, fontSize, lineHeight }}>
          {lines.map((line, index) => {
            if (hidden.includes(index)) {
              return null
            }
            const appear = entrance(frame, appearAt[index] ?? Number.POSITIVE_INFINITY, 10)
            const indent = line.text.length - line.text.trimStart().length
            const lineHighlights = highlights.filter(highlight => highlight.line === index)
            const emphasis = Math.max(0, ...lineHighlights.map(highlight => between(frame, highlight.from, highlight.to ?? Number.POSITIVE_INFINITY)))
            const tone = lineHighlights.find(highlight => between(frame, highlight.from, highlight.to ?? Number.POSITIVE_INFINITY) > 0)?.tone
            const callout = lineHighlights.find(highlight => highlight.callout && between(frame, highlight.from, highlight.to ?? Number.POSITIVE_INFINITY) > 0)
            return (
              <Fragment key={`${index}-${line.seconds}`}>
                {chip && chip.beforeLine === index ? <Chip chip={chip} frame={frame} /> : null}
                <div
                  style={{
                    position: 'relative',
                    display: 'flex',
                    opacity: appear * lineVisibility(highlights, frame, index),
                    transform: `translateX(${(1 - appear) * -12}px)`,
                  }}
                >
                  <div
                    style={{
                      width: textWidth,
                      paddingLeft: `${indent}ch`,
                      whiteSpace: 'pre-wrap',
                      borderRadius: 8,
                      background: tone ? `${toneColour(tone)}${Math.round(emphasis * 0x24).toString(16).padStart(2, '0')}` : 'transparent',
                      boxShadow: tone ? `inset 4px 0 0 ${toneColour(tone)}${Math.round(emphasis * 255).toString(16).padStart(2, '0')}` : 'none',
                    }}
                  >
                    {segments(line.text.trimStart()).map((segment, segmentIndex) => (
                      <span
                        key={segmentIndex}
                        style={{
                          fontWeight: segment.bold ? 700 : 400,
                          color: segment.tone ? toneColour(segment.tone) : segment.dim ? colours.dim : colours.text,
                        }}
                      >
                        {segment.text}
                      </span>
                    ))}
                  </div>
                  {callout?.callout ? (
                    <div
                      style={{
                        position: 'absolute',
                        left: textWidth + 40,
                        top: 2,
                        opacity: emphasis,
                        transform: `translateX(${(1 - emphasis) * 16}px)`,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                        fontFamily: fonts.sans,
                      }}
                    >
                      <span style={{ fontSize: 30, color: toneColour(callout.tone) }}>←</span>
                      <span
                        style={{
                          fontSize: 24,
                          fontWeight: 600,
                          padding: '6px 14px',
                          borderRadius: 999,
                          border: `2px solid ${toneColour(callout.tone)}`,
                          color: toneColour(callout.tone),
                          background: colours.background,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {callout.callout}
                      </span>
                    </div>
                  ) : null}
                </div>
              </Fragment>
            )
          })}
        </div>
      </div>
      <div style={{ marginTop: 14, fontSize: 20, color: colours.dim, opacity: 0.8 }}>{footnote}</div>
    </div>
  )
}
