import { plain } from './ansi'
import { Caption } from './Caption'
import type { DemoProps, Section } from './data'
import { modelName } from './data'
import { Terminal, type Highlight } from './Terminal'
import type { ApproverTiming, CleanTiming, InjectionTiming, Span, TraceTiming, UnclearTiming } from './timings'
import { TraceZoom } from './TraceZoom'

const lineIndex = (section: Section, pattern: RegExp) => {
  const found = section.lines.findIndex(line => pattern.test(plain(line.text)))
  if (found < 0) {
    throw new Error(`No line matching ${pattern} under ${JSON.stringify(plain(section.heading.text))}`)
  }
  return found + 1
}

const linesOf = (section: Section) => [section.heading, ...section.lines]

const footnote = (props: DemoProps) =>
  `Real output from the recorded run on ${props.recordedOn} · waits are sped up and labelled`

type BeatProps<Timing> = { props: DemoProps; section: Section; realSeconds: number; timing: Timing }

const reviewChip = (
  props: DemoProps,
  chip: { at: number; duration: number },
  realSeconds: number,
  beforeLine: number,
) => ({
  beforeLine,
  at: chip.at,
  duration: chip.duration,
  realSeconds,
  label: `Worker review with ${modelName(props.primaryModelId)}`,
})

export const CleanBeat = ({ props, section, realSeconds, timing }: BeatProps<CleanTiming>) => {
  const outcome = lineIndex(section, /AUTO-APPROVED/)
  const highlights: Highlight[] = [{ line: outcome, ...timing.outcome, tone: 'green', callout: 'no human needed' }]
  return (
    <>
      <Caption
        kicker="1 of 4"
        title="A clean, low-value claim"
        beatFrames={timing.frames}
        subtitles={timing.subtitles}
      />
      <Terminal
        title="spendguard · local stack"
        lines={linesOf(section)}
        appearAt={[timing.appear.heading, timing.appear.purpose, timing.appear.outcome]}
        chip={reviewChip(props, timing.chip, realSeconds, outcome)}
        highlights={highlights}
        fontSize={33}
        footnote={footnote(props)}
      />
    </>
  )
}

const appearing = (count: number, first: number, gap: number) =>
  Array.from({ length: count }, (_, index) => first + index * gap)

export const UnclearBeat = ({ props, section, realSeconds, timing }: BeatProps<UnclearTiming>) => {
  const review = lineIndex(section, /NEEDS REVIEW/)
  const quote = lineIndex(section, /^\s*quote\b/)
  const clause = lineIndex(section, /^\s*clause\b/)
  const lines = linesOf(section)
  return (
    <>
      <Caption
        kicker="2 of 4"
        title="A vague business purpose"
        beatFrames={timing.frames}
        subtitles={timing.subtitles}
      />
      <Terminal
        title="spendguard · local stack"
        lines={lines}
        appearAt={[
          timing.appear.heading,
          timing.appear.purpose,
          ...appearing(lines.length - 2, timing.appear.review, timing.appear.gap),
        ]}
        chip={reviewChip(props, timing.chip, realSeconds, review)}
        fontSize={28}
        highlights={[
          { line: 1, ...timing.purpose, tone: 'amber', callout: 'who with?' },
          { line: quote, ...timing.quote, tone: 'amber', callout: 'verbatim quote' },
          { line: clause, ...timing.clause, tone: 'accent', callout: 'the rule' },
        ]}
        footnote={footnote(props)}
      />
    </>
  )
}

export const InjectionBeat = ({ props, section, realSeconds, timing }: BeatProps<InjectionTiming>) => {
  const review = lineIndex(section, /NEEDS REVIEW/)
  const quote = lineIndex(section, /^\s*quote\b/)
  const lines = linesOf(section)
  const hidden = lines.flatMap((line, index) =>
    index > quote && /^\s*(quote|clause)\b/.test(plain(line.text)) ? [index] : [],
  )
  const extraFindings = lines.flatMap((line, index) =>
    index > quote && !hidden.includes(index) ? [index] : [],
  )
  const { extraFindings: extraFindingsSpan } = timing
  return (
    <>
      <Caption
        kicker="3 of 4"
        title="An injection attempt"
        beatFrames={timing.frames}
        subtitles={timing.subtitles}
      />
      <Terminal
        title="spendguard · local stack"
        lines={lines}
        appearAt={[
          timing.appear.heading,
          timing.appear.purpose,
          ...appearing(lines.length - 2, timing.appear.review, timing.appear.gap),
        ]}
        chip={reviewChip(props, timing.chip, realSeconds, review)}
        hidden={hidden}
        fontSize={28}
        highlights={[
          { line: 1, ...timing.attack, tone: 'red', callout: 'the attack' },
          { line: quote, ...timing.quote, tone: 'red', callout: 'quoted as evidence' },
          { line: review, ...timing.review, tone: 'amber', callout: 'not auto-approved' },
          ...(extraFindingsSpan
            ? extraFindings.map((line, index) => ({
                line,
                ...extraFindingsSpan,
                tone: 'accent' as const,
                callout: index === 0 ? 'not really there' : undefined,
              }))
            : []),
        ]}
        footnote={footnote(props)}
      />
    </>
  )
}

export const ApproverBeat = ({ props, section, realSeconds, timing }: BeatProps<ApproverTiming>) => {
  const approve = lineIndex(section, /APPROVE\b/)
  const reject = lineIndex(section, /REJECT\b/)
  const executed = lineIndex(section, /EXECUTED/)
  const precedent = lineIndex(section, /PRECEDENT/)
  const conflict = lineIndex(section, /409 CONFLICT/)
  const lines = linesOf(section)
  const cueFrames = new Map([
    [0, timing.appear.heading],
    [approve, timing.appear.approve],
    [reject, timing.appear.reject],
    [executed, timing.appear.executed],
    [precedent, timing.appear.precedent],
    [conflict, timing.appear.conflict],
  ])
  const appearAt = lines.map((_, index) => cueFrames.get(index) ?? timing.appear.conflict)
  const optional = (line: number, span: Span | undefined, tone: Highlight['tone'], callout: string) =>
    span ? [{ line, ...span, tone, callout }] : []
  return (
    <>
      <Caption
        kicker="4 of 4"
        title="The approver decides"
        beatFrames={timing.frames}
        subtitles={timing.subtitles}
      />
      <Terminal
        title="spendguard · local stack"
        lines={lines}
        appearAt={appearAt}
        chip={{ ...reviewChip(props, timing.chip, realSeconds, executed), label: 'Both decisions executed by the worker' }}
        fontSize={30}
        highlights={[
          ...optional(approve, timing.approve, 'green', 'with a reason'),
          ...optional(reject, timing.reject, 'red', 'with a reason'),
          { line: executed, ...timing.executed, tone: 'green', callout: 'exactly once' },
          { line: precedent, ...timing.precedent, tone: 'cyan', callout: 'informs, never approves' },
          { line: conflict, ...timing.conflict, tone: 'red', callout: 'refused' },
        ]}
        footnote={footnote(props)}
      />
    </>
  )
}

export const TraceBeat = ({ props, timing }: { props: DemoProps; timing: TraceTiming }) => (
  <>
    <Caption
      kicker="Under the hood"
      title="The Langfuse trace of the injected claim"
      beatFrames={timing.frames}
      subtitles={timing.subtitles}
    />
    <TraceZoom
      trace={props.trace}
      cues={[
        {
          shot: 'verify',
          ...timing.verify,
          focus: ['finding', 'quote'],
          scale: 2.1,
          highlights: [{ box: 'quote', tone: 'red', label: 'the injected text, quoted as evidence' }],
        },
        {
          shot: 'route',
          ...timing.route,
          to: Number.POSITIVE_INFINITY,
          focus: ['route'],
          scale: 2.8,
          highlights: [{ box: 'route', tone: 'amber', label: 'human review, not auto-approval', labelSide: 'right' }],
        },
      ]}
    />
  </>
)
