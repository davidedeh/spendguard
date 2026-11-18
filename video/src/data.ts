export type CapturedLine = { seconds: number; text: string }

export type Box = { x: number; y: number; width: number; height: number }

export type Shot = { image: string; boxes: Record<string, Box> }

export type TraceShots = {
  viewport: { width: number; height: number }
  deviceScaleFactor: number
  shots: { verify: Shot; route: Shot }
}

export type DemoProps = {
  lines: CapturedLine[]
  reviewSeconds: number[]
  primaryModelId: string
  recordedOn: string
  repositoryUrl: string
  trace: TraceShots
}

export type Section = { heading: CapturedLine; lines: CapturedLine[] }

export const sections = (lines: CapturedLine[]): Section[] => {
  const found: Section[] = []
  for (const line of lines) {
    const current = found.at(-1)
    if (!line.text.startsWith(' ')) {
      found.push({ heading: line, lines: [] })
    } else if (current) {
      current.lines.push(line)
    }
  }
  return found
}

const modelNames: [string, string][] = [
  ['claude-haiku-4-5', 'Claude Haiku 4.5'],
  ['claude-sonnet-4-6', 'Claude Sonnet 4.6'],
]

export const modelName = (modelId: string) =>
  modelNames.find(([fragment]) => modelId.includes(fragment))?.[1] ?? modelId
