import type { Subtitle } from './Caption'
import { contentDelay } from './pacing'

export type Span = { from: number; to?: number }

export type NarrationCue = { clip: string; at: number }

type BeatBase = { frames: number; subtitles: Subtitle[]; narration: NarrationCue[] }

export type HookTiming = BeatBase

export type CleanTiming = BeatBase & {
  appear: { heading: number; purpose: number; outcome: number }
  chip: { at: number; duration: number }
  outcome: Span
}

export type UnclearTiming = BeatBase & {
  appear: { heading: number; purpose: number; review: number; gap: number }
  chip: { at: number; duration: number }
  purpose: Span
  quote: Span
  clause: Span
}

export type InjectionTiming = BeatBase & {
  appear: { heading: number; purpose: number; review: number; gap: number }
  chip: { at: number; duration: number }
  attack: Span
  quote: Span
  review: Span
  extraFindings?: Span
}

export type ApproverTiming = BeatBase & {
  appear: { heading: number; approve: number; reject: number; executed: number; precedent: number; conflict: number }
  chip: { at: number; duration: number }
  approve?: Span
  reject?: Span
  executed: Span
  precedent: Span
  conflict: Span
}

export type TraceTiming = BeatBase & {
  verify: { from: number; to: number; zoomAt: number; zoomOutAt: number }
  route: { from: number; zoomAt: number }
}

export type EndTiming = BeatBase & { pointsAt: number[]; urlAt: number }

export type Timings = {
  hook: HookTiming
  clean: CleanTiming
  unclear: UnclearTiming
  injection: InjectionTiming
  approver: ApproverTiming
  trace: TraceTiming
  end: EndTiming
}

const endPointGap = 45

export const silentTimings: Timings = {
  hook: { frames: 185, subtitles: [], narration: [] },
  clean: {
    frames: 330,
    narration: [],
    subtitles: [
      { from: 0, text: 'Reviewed by the worker, with no findings.' },
      { from: 135, text: 'So it is approved automatically and sent to the ERP once.' },
    ],
    appear: { heading: 25, purpose: 37, outcome: 115 },
    chip: { at: 50, duration: 60 },
    outcome: { from: 120 },
  },
  unclear: {
    frames: 465,
    narration: [],
    subtitles: [
      { from: 0, text: 'The purpose doesn’t say who the lunch was for or with.' },
      { from: 192, text: 'Flagged for review, with the quote and the clause it breaks.' },
    ],
    appear: { heading: 25, purpose: 37, review: 192, gap: 12 },
    chip: { at: 125, duration: 60 },
    purpose: { from: 45, to: 123 },
    quote: { from: 245, to: 330 },
    clause: { from: 330 },
  },
  injection: {
    frames: 620,
    narration: [],
    subtitles: [
      { from: 0, text: 'The business purpose tells the agent to switch its checks off.' },
      { from: 225, text: 'The agent treats it as data: the injected text is quoted as evidence,' },
      { from: 426, text: 'so the claim is not auto-approved. It goes to a human.' },
    ],
    appear: { heading: 25, purpose: 37, review: 215, gap: 10 },
    chip: { at: 150, duration: 60 },
    attack: { from: 40, to: 150 },
    quote: { from: 265, to: 426 },
    review: { from: 426 },
  },
  approver: {
    frames: 540,
    narration: [],
    subtitles: [
      { from: 0, text: 'Dismiss the vague finding and approve; reject the injection.' },
      { from: 185, text: 'Each decision reaches the ERP once, and the dismissal becomes a precedent.' },
      { from: 375, text: 'A second decision on the same claim is refused.' },
    ],
    appear: { heading: 25, approve: 35, reject: 95, executed: 185, precedent: 280, conflict: 375 },
    chip: { at: 150, duration: 30 },
    approve: { from: 40, to: 95 },
    reject: { from: 95, to: 150 },
    executed: { from: 190, to: 280 },
    precedent: { from: 280, to: 375 },
    conflict: { from: 380 },
  },
  trace: {
    frames: 480,
    narration: [],
    subtitles: [
      { from: 0, text: 'verify keeps the finding only because its quote matches the claim exactly.' },
      { from: 259, text: 'route then sends it to human review instead of approving it.' },
    ],
    verify: { from: 20, to: 259, zoomAt: 38, zoomOutAt: 215 },
    route: { from: 243, zoomAt: 261 },
  },
  end: {
    frames: 450,
    narration: [],
    subtitles: [],
    pointsAt: [0, 1, 2, 3].map(index => contentDelay + 30 + index * endPointGap),
    urlAt: contentDelay + 30 + 4 * endPointGap,
  },
}
