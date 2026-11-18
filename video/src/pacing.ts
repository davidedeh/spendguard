import { fps } from './format'

export const transitionFrames = 21
export const contentDelay = transitionFrames
export const wordsPerSecond = 2
export const holdAfterHighlight = 30
export const subtitleFadeFrames = 6
export const zoomFrames = 32

const wordCount = (text: string) => text.split(/\s+/).filter(word => /\w/.test(word)).length

export const readingFrames = (text: string) => Math.ceil((wordCount(text) / wordsPerSecond) * fps)

export const assertReadable = (text: string, frames: number, where: string) => {
  const needed = readingFrames(text)
  if (frames < needed) {
    throw new Error(
      `${where}: ${JSON.stringify(text)} is on screen for ${frames} frames but needs ${needed} at ${wordsPerSecond} words a second`,
    )
  }
}
