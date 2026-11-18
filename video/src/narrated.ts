import type { Subtitle } from './Caption'
import { fps } from './format'
import { entranceFrames } from './motion'
import {
  contentDelay,
  holdAfterHighlight,
  readingFrames,
  subtitleFadeFrames,
  transitionFrames,
  zoomFrames,
} from './pacing'
import type { NarrationCue, Timings } from './timings'

export type NarrationWord = { text: string; seconds: number }

export type NarrationClip = { file: string; seconds: number; words: NarrationWord[] }

export type Narration = Record<string, NarrationClip>

const highlightLandFrames = entranceFrames
const narrationTailFrames = 15
const narrationStart = contentDelay + 3
const hookNarrationStart = 10
const endHoldFrames = 90
const chipFrames = 30
const lineGap = 8

const normalised = (text: string) => text.toLowerCase().replace(/[^a-z]/g, '')

export const clipNamed = (narration: Narration | null, name: string) => {
  const clip = narration?.[name]
  if (!clip) {
    throw new Error(`Narration clip ${JSON.stringify(name)} is missing`)
  }
  return clip
}

const anchoredWord = (narration: Narration, clip: string, anchor: string, fromIndex: number) => {
  const words = clipNamed(narration, clip).words
  const matches = (word: NarrationWord) => normalised(word.text).startsWith(anchor)
  const index = words.findIndex((word, position) => position >= fromIndex && matches(word))
  const word = words[index]
  if (!word) {
    const where = words.some(candidate => matches(candidate)) ? ' after the previous anchor' : ''
    throw new Error(`No word starting ${JSON.stringify(anchor)}${where} in narration clip ${JSON.stringify(clip)}`)
  }
  return { index, frames: Math.round(word.seconds * fps) }
}

const heldUntil = (highlightFrom: number) => highlightFrom + highlightLandFrames + holdAfterHighlight

class Track {
  readonly subtitles: Subtitle[] = []
  private readonly placed: { cue: NarrationCue; nextAnchorFrom: number }[] = []
  private nextCaptionAt = 0
  private nextHighlightAt = 0
  private lastEnd = 0

  constructor(readonly narration: Narration) {}

  get clips() {
    return this.placed.map(entry => entry.cue)
  }

  place(clip: string, at: number) {
    this.placed.push({ cue: { clip, at }, nextAnchorFrom: 0 })
    return this
  }

  private placedClip(clip: string) {
    const entry = this.placed.find(candidate => candidate.cue.clip === clip)
    if (!entry) {
      throw new Error(`Narration clip ${JSON.stringify(clip)} is not placed in this beat`)
    }
    return entry
  }

  word(clip: string, anchor: string) {
    const entry = this.placedClip(clip)
    const match = anchoredWord(this.narration, clip, anchor, entry.nextAnchorFrom)
    entry.nextAnchorFrom = match.index
    return entry.cue.at + match.frames
  }

  clipEnd(clip: string) {
    return this.placedClip(clip).cue.at + Math.round(clipNamed(this.narration, clip).seconds * fps)
  }

  caption(text: string, wanted: number) {
    const from = Math.max(wanted, this.nextCaptionAt)
    const end = Math.max(from, contentDelay) + subtitleFadeFrames + readingFrames(text)
    this.subtitles.push({ from, text })
    this.nextCaptionAt = end
    this.lastEnd = Math.max(this.lastEnd, end)
    return from
  }

  highlight(wanted: number) {
    const from = Math.max(wanted, this.nextHighlightAt)
    this.nextHighlightAt = heldUntil(from)
    this.lastEnd = Math.max(this.lastEnd, this.nextHighlightAt)
    return from
  }

  holdUntil(frame: number) {
    this.lastEnd = Math.max(this.lastEnd, frame)
  }

  frames(last = false) {
    const narrationEnd = Math.max(...this.clips.map(cue => this.clipEnd(cue.clip))) + narrationTailFrames
    return Math.max(this.lastEnd, narrationEnd) + (last ? 0 : transitionFrames)
  }
}

export const narratedTimings = (narration: Narration): Timings => {
  const hook = new Track(narration).place('1', hookNarrationStart)

  const clean = new Track(narration).place('2', narrationStart)
  clean.caption('First, a clean, low-value claim.', 0)
  const cleanChipAt = clean.word('2', 'rule')
  clean.caption('Rule checks pass; the model finds nothing,', cleanChipAt)
  const cleanOutcomeAt = clean.word('2', 'approved')
  const cleanOutcome = clean.highlight(cleanOutcomeAt)
  clean.caption('so it’s approved and sent to the ERP once.', cleanOutcomeAt)

  const unclear = new Track(narration).place('3', narrationStart)
  unclear.caption('Next, a business purpose that’s too vague.', 0)
  const unclearPurpose = unclear.highlight(unclear.word('3', 'vague'))
  const unclearChipAt = Math.max(unclear.word('3', 'model'), heldUntil(unclearPurpose))
  unclear.caption('The model flags it, quoting the text', unclear.word('3', 'model'))
  const unclearReviewAt = unclearChipAt + chipFrames
  const unclearQuote = unclear.highlight(Math.max(unclear.word('3', 'quote'), unclearReviewAt + 2 * lineGap + 4))
  unclear.caption('and naming the clause.', unclear.word('3', 'name'))
  const unclearClause = unclear.highlight(Math.max(unclear.word('3', 'clause'), unclearReviewAt + 3 * lineGap + 4))
  unclear.caption('Code checks both before a human sees it.', unclear.word('3', 'code'))

  const injection = new Track(narration).place('4a', narrationStart)
  injection.caption('The purpose tries to switch the agent’s checks off.', 0)
  const injectionAttack = injection.highlight(injection.word('4a', 'tries'))
  const injectionChipAt = Math.max(injection.word('4a', 'treats'), heldUntil(injectionAttack))
  injection.caption('Treated as data, not instructions:', injection.word('4a', 'treats'))
  const injectionReviewAt = injectionChipAt + chipFrames
  const injectionQuote = injection.highlight(Math.max(injection.word('4a', 'quoted'), injectionReviewAt + 2 * lineGap + 4))
  injection.caption('the injected text is quoted as evidence,', injection.word('4a', 'quoted'))
  const injectionReview = injection.highlight(injection.word('4a', 'human'))
  const lastMainCaption = injection.caption('and it goes to a human.', injection.word('4a', 'human'))
  const asideCaptionText = 'The model also raised findings that aren’t there.'
  const asideAt = Math.max(
    injection.clipEnd('4a') + narrationTailFrames,
    lastMainCaption + subtitleFadeFrames + readingFrames('and it goes to a human.'),
  )
  const asideOffset = anchoredWord(narration, '4b', 'raised', 0).frames
  injection.place('4b', Math.max(asideAt - asideOffset, injection.clipEnd('4a')))
  const injectionExtras = injection.highlight(injection.word('4b', 'raised'))
  injection.caption(asideCaptionText, injection.word('4b', 'raised'))
  injection.caption('Over-flagging only sends a claim to a person,', injection.word('4b', 'fine'))
  injection.caption('and the eval measures how often.', injection.word('4b', 'measures'))

  const approver = new Track(narration).place('5', narrationStart)
  approver.caption('The approver decides.', 0)
  approver.caption('Each decision executes once, even on redelivery,', approver.word('5', 'each'))
  const approverExecuted = approver.highlight(approver.word('5', 'executed'))
  approver.caption('and a second decision is refused.', approver.word('5', 'second'))
  const approverConflict = approver.highlight(approver.word('5', 'refused'))
  const approverPrecedent = approver.highlight(approver.word('5', 'dism'))
  approver.caption('A dismissal with a reason becomes a precedent.', approver.word('5', 'dism'))

  const trace = new Track(narration).place('6', narrationStart)
  trace.caption('Every run is traced in Langfuse.', 0)
  const verifyMarked = trace.highlight(trace.word('6', 'verify'))
  trace.caption('verify keeps it only because the quote matches exactly,', trace.word('6', 'verify'))
  const verifyZoomOutAt = Math.max(trace.word('6', 'matches'), heldUntil(verifyMarked))
  const routeFrom = verifyZoomOutAt + zoomFrames - 2
  const routeZoomAt = Math.max(trace.word('6', 'route') - zoomFrames, routeFrom + 18)
  trace.highlight(routeZoomAt + zoomFrames)
  trace.caption('and route sends it to human review.', trace.word('6', 'route'))

  const end = new Track(narration).place('7', narrationStart)
  const pointsAt = [end.word('7', 'rules'), end.word('7', 'every'), end.word('7', 'human'), end.word('7', 'frozen')]
  const urlAt = end.clipEnd('7')
  end.holdUntil(urlAt + endHoldFrames)

  return {
    hook: { frames: hook.frames(), subtitles: [], narration: hook.clips },
    clean: {
      frames: clean.frames(),
      subtitles: clean.subtitles,
      narration: clean.clips,
      appear: { heading: 25, purpose: 37, outcome: cleanOutcomeAt },
      chip: { at: cleanChipAt, duration: Math.max(chipFrames, cleanOutcomeAt - cleanChipAt - 10) },
      outcome: { from: cleanOutcome },
    },
    unclear: {
      frames: unclear.frames(),
      subtitles: unclear.subtitles,
      narration: unclear.clips,
      appear: { heading: 25, purpose: 37, review: unclearReviewAt, gap: lineGap },
      chip: { at: unclearChipAt, duration: chipFrames },
      purpose: { from: unclearPurpose, to: unclearChipAt },
      quote: { from: unclearQuote, to: unclearClause },
      clause: { from: unclearClause },
    },
    injection: {
      frames: injection.frames(),
      subtitles: injection.subtitles,
      narration: injection.clips,
      appear: { heading: 25, purpose: 37, review: injectionReviewAt, gap: lineGap },
      chip: { at: injectionChipAt, duration: chipFrames },
      attack: { from: injectionAttack, to: injectionChipAt },
      quote: { from: injectionQuote, to: injectionReview },
      review: { from: injectionReview, to: injectionExtras },
      extraFindings: { from: injectionExtras },
    },
    approver: {
      frames: approver.frames(),
      subtitles: approver.subtitles,
      narration: approver.clips,
      appear: { heading: 25, approve: 30, reject: 40, executed: approverExecuted, precedent: heldUntil(approverExecuted), conflict: approverConflict },
      chip: { at: 45, duration: chipFrames },
      executed: { from: approverExecuted, to: approverConflict },
      conflict: { from: approverConflict, to: approverPrecedent },
      precedent: { from: approverPrecedent },
    },
    trace: {
      frames: trace.frames(),
      subtitles: trace.subtitles,
      narration: trace.clips,
      verify: { from: 0, to: routeFrom + 14, zoomAt: verifyMarked - zoomFrames, zoomOutAt: verifyZoomOutAt },
      route: { from: routeFrom, zoomAt: routeZoomAt },
    },
    end: { frames: end.frames(true), subtitles: [], narration: end.clips, pointsAt, urlAt },
  }
}
