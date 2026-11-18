import { useCurrentFrame } from 'remotion'

import { between, entrance } from './motion'
import { assertReadable, contentDelay, subtitleFadeFrames, transitionFrames } from './pacing'
import { colours } from './theme'

export type Subtitle = { from: number; text: string }

export const Caption = ({
  kicker,
  title,
  subtitles,
  beatFrames,
}: {
  kicker: string
  title: string
  subtitles: Subtitle[]
  beatFrames: number
}) => {
  const frame = useCurrentFrame()
  const starts = subtitles.map(subtitle => Math.max(subtitle.from, contentDelay) + subtitleFadeFrames)
  subtitles.forEach((subtitle, index) => {
    const end = subtitles[index + 1]?.from ?? beatFrames - transitionFrames
    assertReadable(subtitle.text, end - (starts[index] ?? 0), title)
  })
  const shown = entrance(frame, contentDelay, 16)
  return (
    <div style={{ position: 'absolute', left: 120, right: 120, top: 70 }}>
      <div
        style={{
          opacity: shown,
          transform: `translateY(${(1 - shown) * 16}px)`,
          display: 'flex',
          alignItems: 'baseline',
          gap: 24,
        }}
      >
        <span
          style={{
            color: colours.accent,
            fontSize: 30,
            fontWeight: 700,
            letterSpacing: 2,
            textTransform: 'uppercase',
          }}
        >
          {kicker}
        </span>
        <span style={{ fontSize: 54, fontWeight: 700, letterSpacing: -0.5 }}>{title}</span>
      </div>
      <div style={{ position: 'relative', height: 50, marginTop: 14 }}>
        {subtitles.map((subtitle, index) => {
          const next = subtitles[index + 1]
          const visible = between(frame, starts[index] ?? 0, next ? next.from : Number.POSITIVE_INFINITY, 12)
          return (
            <div
              key={subtitle.text}
              style={{
                position: 'absolute',
                fontSize: 36,
                fontWeight: 500,
                color: colours.dim,
                opacity: visible,
                transform: `translateY(${(1 - visible) * 10}px)`,
              }}
            >
              {subtitle.text}
            </div>
          )
        })}
      </div>
    </div>
  )
}
