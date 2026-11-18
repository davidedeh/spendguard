import { Easing, interpolate } from 'remotion'

export const easeInOut = Easing.bezier(0.65, 0, 0.35, 1)
export const easeOut = Easing.bezier(0.16, 1, 0.3, 1)

export const entranceFrames = 14

export const entrance = (frame: number, at: number, duration = entranceFrames) =>
  Number.isFinite(at)
    ? interpolate(frame, [at, at + duration], [0, 1], {
        easing: easeOut,
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 0

export const between = (frame: number, from: number, to: number, duration = entranceFrames) =>
  Math.min(entrance(frame, from, duration), 1 - entrance(frame, to, duration))
