import type { ReactNode } from 'react'
import { AbsoluteFill, interpolate, useCurrentFrame } from 'remotion'

import { easeInOut } from './motion'
import { transitionFrames } from './pacing'

export const FadingBeat = ({ frames, last, children }: { frames: number; last: boolean; children: ReactNode }) => {
  const frame = useCurrentFrame()
  const opacity = last
    ? 1
    : interpolate(frame, [frames - transitionFrames, frames], [1, 0], {
        easing: easeInOut,
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>
}
