import { fade } from '@remotion/transitions/fade'
import { linearTiming, TransitionSeries } from '@remotion/transitions'
import { Audio, Sequence, staticFile } from 'remotion'

import { plain } from './ansi'
import { Backdrop } from './Backdrop'
import { ApproverBeat, CleanBeat, InjectionBeat, TraceBeat, UnclearBeat } from './beats'
import { type DemoProps, modelName, sections } from './data'
import { EndCard } from './EndCard'
import { FadingBeat } from './FadingBeat'
import { Hook } from './Hook'
import { easeInOut } from './motion'
import { clipNamed, type Narration } from './narrated'
import { transitionFrames } from './pacing'
import type { NarrationCue, Timings } from './timings'

const seconds = (value: number) => Math.round(value * 10) / 10

export const totalFrames = (timings: Timings) => {
  const beats = Object.values(timings)
  return beats.reduce((sum, beat) => sum + beat.frames, 0) - (beats.length - 1) * transitionFrames
}

export type VideoProps = DemoProps & { timings: Timings; narration: Narration | null }

const NarrationAudio = ({ cues, narration }: { cues: NarrationCue[]; narration: Narration | null }) => (
  <>
    {cues.map(cue => {
      const clip = clipNamed(narration, cue.clip)
      return (
        <Sequence key={cue.clip} from={cue.at} layout="none">
          <Audio src={staticFile(`take/narration/${clip.file}`)} />
        </Sequence>
      )
    })}
  </>
)

export const Video = (videoProps: VideoProps) => {
  const { timings, narration, ...props } = videoProps
  const [clean, unclear, injection, approver] = sections(props.lines)
  if (!clean || !unclear || !injection || !approver) {
    throw new Error(`Expected four sections, got ${JSON.stringify(sections(props.lines).map(section => plain(section.heading.text)))}`)
  }
  const [cleanSeconds, unclearSeconds, injectionSeconds] = props.reviewSeconds
  if (cleanSeconds === undefined || unclearSeconds === undefined || injectionSeconds === undefined) {
    throw new Error(`Expected three review durations, got ${JSON.stringify(props.reviewSeconds)}`)
  }
  const executed = approver.lines.find(line => plain(line.text).includes('EXECUTED'))
  const rejected = approver.lines.find(line => plain(line.text).includes('REJECT'))
  if (!executed || !rejected) {
    throw new Error('The approver section has no REJECT or EXECUTED line')
  }
  const stack = [
    'LangGraph',
    `${modelName(props.primaryModelId)} · Bedrock`,
    'FastAPI',
    'Postgres + pgvector',
    'SQS',
    'Langfuse',
  ]
  const beats = [
    { key: 'hook', timing: timings.hook, element: <Hook stack={stack} frames={timings.hook.frames} /> },
    {
      key: 'clean',
      timing: timings.clean,
      element: <CleanBeat props={props} section={clean} realSeconds={cleanSeconds} timing={timings.clean} />,
    },
    {
      key: 'unclear',
      timing: timings.unclear,
      element: <UnclearBeat props={props} section={unclear} realSeconds={unclearSeconds} timing={timings.unclear} />,
    },
    {
      key: 'injection',
      timing: timings.injection,
      element: (
        <InjectionBeat props={props} section={injection} realSeconds={injectionSeconds} timing={timings.injection} />
      ),
    },
    {
      key: 'approver',
      timing: timings.approver,
      element: (
        <ApproverBeat
          props={props}
          section={approver}
          realSeconds={seconds(executed.seconds - rejected.seconds)}
          timing={timings.approver}
        />
      ),
    },
    { key: 'trace', timing: timings.trace, element: <TraceBeat props={props} timing={timings.trace} /> },
    { key: 'end', timing: timings.end, element: <EndCard repositoryUrl={props.repositoryUrl} timing={timings.end} /> },
  ]
  return (
    <Backdrop>
      <TransitionSeries>
        {beats.flatMap((beat, index) => [
          ...(index === 0
            ? []
            : [
                <TransitionSeries.Transition
                  key={`${beat.key}-transition`}
                  presentation={fade()}
                  timing={linearTiming({ durationInFrames: transitionFrames, easing: easeInOut })}
                />,
              ]),
          <TransitionSeries.Sequence key={beat.key} durationInFrames={beat.timing.frames}>
            <FadingBeat frames={beat.timing.frames} last={index === beats.length - 1}>
              {beat.element}
            </FadingBeat>
            <NarrationAudio cues={beat.timing.narration} narration={narration} />
          </TransitionSeries.Sequence>,
        ])}
      </TransitionSeries>
    </Backdrop>
  )
}
