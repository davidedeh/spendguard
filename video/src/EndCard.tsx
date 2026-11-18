import { AbsoluteFill, useCurrentFrame } from 'remotion'

import { entrance } from './motion'
import { assertReadable, contentDelay } from './pacing'
import { colours } from './theme'
import type { EndTiming } from './timings'

const points = [
  ['Rules vs judgement', 'Code before model.'],
  ['Deterministic verify', 'Every finding checked.'],
  ['Durable human review', 'Decisions execute once.'],
  ['Measured evals', 'A frozen eval set.'],
] as const

export const EndCard = ({ repositoryUrl, timing }: { repositoryUrl: string; timing: EndTiming }) => {
  const { frames } = timing
  const text = ['How SpendGuard is built', ...points.flat(), repositoryUrl].join(' ')
  assertReadable(text, frames - contentDelay, 'End card')
  const frame = useCurrentFrame()
  const heading = entrance(frame, contentDelay, 16)
  const url = entrance(frame, timing.urlAt, 16)
  return (
    <AbsoluteFill style={{ padding: '0 160px', justifyContent: 'center' }}>
      <div style={{ opacity: heading, fontSize: 72, fontWeight: 700, letterSpacing: -1 }}>
        How Spend<span style={{ color: colours.accent }}>Guard</span> is built
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, marginTop: 64 }}>
        {points.map(([title, detail], index) => {
          const shown = entrance(frame, timing.pointsAt[index] ?? timing.urlAt, 16)
          return (
            <div
              key={title}
              style={{
                opacity: shown,
                transform: `translateY(${(1 - shown) * 16}px)`,
                padding: '40px 44px',
                borderRadius: 20,
                background: colours.panel,
                border: `1px solid ${colours.panelBorder}`,
              }}
            >
              <div style={{ fontSize: 44, fontWeight: 700, color: colours.accent }}>{title}</div>
              <div style={{ fontSize: 34, fontWeight: 500, marginTop: 12, lineHeight: 1.35, color: colours.text }}>{detail}</div>
            </div>
          )
        })}
      </div>
      <div style={{ opacity: url, marginTop: 64, fontSize: 44, fontWeight: 600, color: colours.dim }}>
        {repositoryUrl.replace(/^https:\/\//, '')}
      </div>
    </AbsoluteFill>
  )
}
