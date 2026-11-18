import type { CSSProperties } from 'react'

import { colours, fonts } from './theme'

export const WindowTitleBar = ({ title, style }: { title: string; style: CSSProperties }) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      fontSize: 20,
      color: colours.dim,
      ...style,
    }}
  >
    {[colours.red, colours.amber, colours.green].map(dot => (
      <span key={dot} style={{ width: 13, height: 13, borderRadius: 7, background: dot, opacity: 0.8 }} />
    ))}
    <span style={{ marginLeft: 14, fontFamily: fonts.mono }}>{title}</span>
  </div>
)
