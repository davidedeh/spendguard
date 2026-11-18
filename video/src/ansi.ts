import type { Tone } from './theme'

export type Segment = { text: string; bold: boolean; dim: boolean; tone: Tone | null }

const escapePattern = /\x1b\[([0-9;]*)m/g

const tones: Record<string, Tone> = { '31': 'red', '32': 'green', '33': 'amber', '36': 'cyan' }

export const segments = (text: string): Segment[] => {
  const found: Segment[] = []
  let style: Omit<Segment, 'text'> = { bold: false, dim: false, tone: null }
  let position = 0
  for (const match of text.matchAll(escapePattern)) {
    if (match.index > position) {
      found.push({ ...style, text: text.slice(position, match.index) })
    }
    for (const code of (match[1] ?? '').split(';')) {
      if (code === '' || code === '0') {
        style = { bold: false, dim: false, tone: null }
      } else if (code === '1') {
        style = { ...style, bold: true }
      } else if (code === '2') {
        style = { ...style, dim: true }
      } else if (Object.hasOwn(tones, code)) {
        style = { ...style, tone: tones[code] ?? null }
      }
    }
    position = match.index + match[0].length
  }
  if (position < text.length) {
    found.push({ ...style, text: text.slice(position) })
  }
  return found
}

export const plain = (text: string) => text.replace(escapePattern, '')
