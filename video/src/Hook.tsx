import { AbsoluteFill, useCurrentFrame } from 'remotion'

import { entrance } from './motion'
import { assertReadable, transitionFrames } from './pacing'
import { colours } from './theme'

const lineAt = 10
const line = 'Checks expense claims against company policy, with a human approver.'

export const Hook = ({ stack, frames }: { stack: string[]; frames: number }) => {
  assertReadable(line, frames - transitionFrames - lineAt, 'Title card')
  const frame = useCurrentFrame()
  const title = entrance(frame, 0, 18)
  const lineShown = entrance(frame, lineAt, 18)
  return (
    <AbsoluteFill style={{ justifyContent: 'center', padding: '0 160px' }}>
      <div style={{ opacity: title, transform: `translateY(${(1 - title) * 20}px)`, fontSize: 150, fontWeight: 700, letterSpacing: -4 }}>
        Spend<span style={{ color: colours.accent }}>Guard</span>
      </div>
      <div
        style={{
          opacity: lineShown,
          transform: `translateY(${(1 - lineShown) * 16}px)`,
          marginTop: 18,
          fontSize: 56,
          fontWeight: 500,
          lineHeight: 1.3,
          maxWidth: 1550,
          textWrap: 'balance',
        }}
      >
        {line}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginTop: 56 }}>
        {stack.map((item, index) => {
          const shown = entrance(frame, 34 + index * 4, 14)
          return (
            <span
              key={item}
              style={{
                opacity: shown,
                transform: `translateY(${(1 - shown) * 12}px)`,
                fontSize: 30,
                fontWeight: 600,
                padding: '12px 24px',
                borderRadius: 999,
                border: `2px solid ${colours.panelBorder}`,
                background: colours.panel,
                color: colours.text,
              }}
            >
              {item}
            </span>
          )
        })}
      </div>
    </AbsoluteFill>
  )
}
