import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from 'remotion'

import type { Box, Shot, TraceShots } from './data'
import { between, easeInOut, entrance } from './motion'
import { contentDelay, zoomFrames } from './pacing'
import { colours, fonts, type Tone, withAlpha } from './theme'
import { WindowTitleBar } from './WindowTitleBar'

export type ZoomCue = {
  shot: keyof TraceShots['shots']
  from: number
  to: number
  focus: string[]
  scale: number
  zoomAt: number
  zoomOutAt?: number
  highlights: { box: string; tone: Tone; label?: string; labelSide?: 'below' | 'right' }[]
}

const stage = { left: 120, top: 300, width: 1680, height: 720 }
const screenPadding = 10
const focusMargin = 110

const fitScale = (trace: TraceShots) =>
  Math.min(stage.width / trace.viewport.width, stage.height / trace.viewport.height)

const boxNamed = (shot: Shot, shotName: string, name: string) => {
  const box = shot.boxes[name]
  if (!box) {
    throw new Error(`No box ${JSON.stringify(name)} in ${JSON.stringify(shotName)}`)
  }
  return box
}

const union = (boxes: Box[]): Box => {
  const left = Math.min(...boxes.map(box => box.x))
  const top = Math.min(...boxes.map(box => box.y))
  const right = Math.max(...boxes.map(box => box.x + box.width))
  const bottom = Math.max(...boxes.map(box => box.y + box.height))
  return { x: left, y: top, width: right - left, height: bottom - top }
}

const centre = (box: Box) => ({ x: box.x + box.width / 2, y: box.y + box.height / 2 })

const ShotView = ({ trace, shot, cue, frame }: { trace: TraceShots; shot: Shot; cue: ZoomCue; frame: number }) => {
  const base = fitScale(trace)
  const focused = union(cue.focus.map(name => boxNamed(shot, cue.shot, name)))
  const focusBox = { ...focused, x: focused.x - focusMargin, width: focused.width + focusMargin * 2 }
  const zoomIn = interpolate(frame, [cue.zoomAt, cue.zoomAt + zoomFrames], [0, 1], {
    easing: easeInOut,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  })
  const zoomOut =
    cue.zoomOutAt === undefined
      ? 0
      : interpolate(frame, [cue.zoomOutAt, cue.zoomOutAt + zoomFrames], [0, 1], {
          easing: easeInOut,
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        })
  const zoom = zoomIn - zoomOut
  const fittedScale = Math.min(cue.scale, stage.width / focusBox.width)
  const scale = base + (fittedScale - base) * zoom
  const page = { x: trace.viewport.width / 2, y: trace.viewport.height / 2 }
  const target = centre(focusBox)
  const focus = { x: page.x + (target.x - page.x) * zoom, y: page.y + (target.y - page.y) * zoom }
  const translate = {
    x: stage.width / 2 - focus.x * scale,
    y: stage.height / 2 - focus.y * scale,
  }
  const toStage = (box: Box) => ({
    left: translate.x + box.x * scale - screenPadding,
    top: translate.y + box.y * scale - screenPadding / 2,
    width: box.width * scale + screenPadding * 2,
    height: box.height * scale + screenPadding,
  })
  const marked = Math.min(
    entrance(frame, cue.zoomAt + zoomFrames, 14),
    1 - entrance(frame, cue.zoomOutAt ?? Number.POSITIVE_INFINITY, 10),
  )
  return (
    <AbsoluteFill style={{ opacity: between(frame, cue.from, cue.to, 14) }}>
      <Img
        src={staticFile(`take/${shot.image}`)}
        style={{
          position: 'absolute',
          left: translate.x,
          top: translate.y,
          width: trace.viewport.width * scale,
          height: trace.viewport.height * scale,
        }}
      />
      <AbsoluteFill style={{ background: withAlpha(colours.background, 0.16 * marked) }} />
      {cue.highlights.map(highlight => {
        const box = boxNamed(shot, cue.shot, highlight.box)
        const position = toStage(box)
        return (
          <div key={highlight.box}>
            <div
              style={{
                position: 'absolute',
                ...position,
                borderRadius: 10,
                border: `4px solid ${colours[highlight.tone]}`,
                background: 'rgba(255, 255, 255, 0.9)',
                overflow: 'hidden',
                opacity: marked,
                boxShadow: `0 0 0 4px ${colours.background}55, 0 0 40px ${colours[highlight.tone]}88`,
              }}
            >
              <Img
                src={staticFile(`take/${shot.image}`)}
                style={{
                  position: 'absolute',
                  left: translate.x - position.left - 4,
                  top: translate.y - position.top - 4,
                  width: trace.viewport.width * scale,
                  height: trace.viewport.height * scale,
                  borderRadius: 6,
                }}
              />
            </div>
            {highlight.label ? (
              <div
                style={{
                  position: 'absolute',
                  left: highlight.labelSide === 'right' ? position.left + position.width + 18 : position.left,
                  top:
                    highlight.labelSide === 'right'
                      ? position.top + position.height / 2 - 22
                      : position.top + position.height + 14,
                  opacity: marked,
                  transform: `translateY(${(1 - marked) * 10}px)`,
                  fontFamily: fonts.sans,
                  fontSize: 26,
                  fontWeight: 600,
                  padding: '8px 16px',
                  borderRadius: 999,
                  color: colours.background,
                  background: colours[highlight.tone],
                }}
              >
                {highlight.label}
              </div>
            ) : null}
          </div>
        )
      })}
    </AbsoluteFill>
  )
}

export const TraceZoom = ({ trace, cues }: { trace: TraceShots; cues: ZoomCue[] }) => {
  const frame = useCurrentFrame()
  const shown = entrance(frame, contentDelay, 18)
  return (
    <div
      style={{
        position: 'absolute',
        left: stage.left,
        top: stage.top - 52,
        width: stage.width,
        opacity: shown,
        transform: `translateY(${(1 - shown) * 24}px)`,
      }}
    >
      <WindowTitleBar
        title="cloud.langfuse.com · public trace"
        style={{
          height: 52,
          padding: '0 22px',
          background: colours.panel,
          border: `1px solid ${colours.panelBorder}`,
          borderBottom: 'none',
          borderRadius: '18px 18px 0 0',
        }}
      />
      <div
        style={{
          position: 'relative',
          width: stage.width,
          height: stage.height,
          overflow: 'hidden',
          background: '#FFFFFF',
          border: `1px solid ${colours.panelBorder}`,
          borderRadius: '0 0 18px 18px',
          boxShadow: '0 30px 80px rgba(0, 0, 0, 0.45)',
        }}
      >
        {cues.map(cue => {
          const shot = trace.shots[cue.shot]
          return <ShotView key={`${cue.shot}-${cue.from}`} trace={trace} shot={shot} cue={cue} frame={frame} />
        })}
      </div>
    </div>
  )
}
