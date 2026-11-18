import type { ReactNode } from 'react'
import { AbsoluteFill } from 'remotion'

import { colours, fonts } from './theme'

export const Backdrop = ({ children }: { children: ReactNode }) => (
  <AbsoluteFill
    style={{
      background: `radial-gradient(1400px 800px at 20% 0%, ${colours.backgroundGlow}, ${colours.background} 70%)`,
      color: colours.text,
      fontFamily: fonts.sans,
    }}
  >
    {children}
  </AbsoluteFill>
)
