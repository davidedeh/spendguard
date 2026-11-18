import { Composition, staticFile } from 'remotion'

import type { DemoProps } from './data'
import { type Narration, narratedTimings } from './narrated'
import { fps, height, width } from './format'
import { silentTimings } from './timings'
import { totalFrames, Video, type VideoProps } from './Video'
import './theme'

const emptyProps: VideoProps = {
  lines: [],
  reviewSeconds: [],
  primaryModelId: '',
  recordedOn: '',
  repositoryUrl: '',
  trace: {
    viewport: { width: 1, height: 1 },
    deviceScaleFactor: 1,
    shots: {
      verify: { image: '', boxes: {} },
      route: { image: '', boxes: {} },
    },
  },
  timings: silentTimings,
  narration: null,
}

const takeProps = async (props: VideoProps): Promise<DemoProps> =>
  props.lines.length > 0 ? props : ((await (await fetch(staticFile('take/props.json'))).json()) as DemoProps)

export const Root = () => (
  <>
    <Composition
      id="SpendGuardDemo"
      component={Video}
      durationInFrames={totalFrames(silentTimings)}
      fps={fps}
      width={width}
      height={height}
      defaultProps={emptyProps}
      calculateMetadata={async ({ props }) => ({
        props: { ...(await takeProps(props)), timings: silentTimings, narration: null },
        durationInFrames: totalFrames(silentTimings),
      })}
    />
    <Composition
      id="SpendGuardDemoNarrated"
      component={Video}
      durationInFrames={totalFrames(silentTimings)}
      fps={fps}
      width={width}
      height={height}
      defaultProps={emptyProps}
      calculateMetadata={async ({ props }) => {
        const narration = (await (await fetch(staticFile('take/narration/narration.json'))).json()) as Narration
        const timings = narratedTimings(narration)
        return {
          props: { ...(await takeProps(props)), timings, narration },
          durationInFrames: totalFrames(timings),
        }
      }}
    />
  </>
)
