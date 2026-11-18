import { loadFont } from '@remotion/fonts'

import inter500 from '@fontsource/inter/files/inter-latin-500-normal.woff2'
import inter600 from '@fontsource/inter/files/inter-latin-600-normal.woff2'
import inter700 from '@fontsource/inter/files/inter-latin-700-normal.woff2'
import mono400 from '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2'
import mono700 from '@fontsource/jetbrains-mono/files/jetbrains-mono-latin-700-normal.woff2'


export const colours = {
  background: '#0B1020',
  backgroundGlow: '#18204A',
  panel: '#111831',
  panelBorder: '#26305C',
  text: '#E7EAF6',
  dim: '#8E97B8',
  accent: '#8B93FF',
  green: '#34D399',
  amber: '#FBBF24',
  red: '#F87171',
  cyan: '#67E8F9',
} as const

export type Tone = 'accent' | 'green' | 'amber' | 'red' | 'cyan'

export const fonts = {
  sans: 'Inter, sans-serif',
  mono: '"JetBrains Mono", monospace',
} as const


const fontFaces = [
  { family: 'Inter', url: inter500, weight: '500' },
  { family: 'Inter', url: inter600, weight: '600' },
  { family: 'Inter', url: inter700, weight: '700' },
  { family: 'JetBrains Mono', url: mono400, weight: '400' },
  { family: 'JetBrains Mono', url: mono700, weight: '700' },
]

for (const face of fontFaces) {
  void loadFont({ family: face.family, url: face.url, weight: face.weight })
}

export const withAlpha = (colour: string, alpha: number) => {
  const [red, green, blue] = [1, 3, 5].map(start => Number.parseInt(colour.slice(start, start + 2), 16))
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}
