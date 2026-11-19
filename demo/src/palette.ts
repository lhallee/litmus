// The litmus scale and the two grounds, taken from site/style.css (dark theme, with paper for contrast beats).

export const brand = {
  human: '#2F6FDB',
  mixed: '#8E5BC9',
  ai: '#D9483B',
  band: '#C9B8E6',
};

export interface Ground {
  ground: string;
  surface: string;
  ink: string;
  soft: string;
  faint: string;
  rule: string;
  humanInk: string;
  mixedInk: string;
  aiInk: string;
  bandWash: string;
  bandEdge: string;
  markRing: string;
  pendingGround: string;
  pendingInk: string;
  unreacted: string;
}

export const dark: Ground = {
  ground: '#16141A',
  surface: '#201D25',
  ink: '#EEE8DC',
  soft: '#B9B2A8',
  faint: '#7A7380',
  rule: '#3A3540',
  humanInk: '#8DB2F5',
  mixedInk: '#C3A3EC',
  aiInk: '#F2907F',
  bandWash: 'rgba(201, 184, 230, 0.62)',
  bandEdge: '#E6DDF3',
  markRing: '#16141A',
  pendingGround: '#2A2630',
  pendingInk: '#E3C98A',
  unreacted: '#E4DCCB',
};

export const paper: Ground = {
  ground: '#F3EEE4',
  surface: '#FBF8F2',
  ink: '#1E1B22',
  soft: '#4F4A55',
  faint: '#8A8290',
  rule: '#D6CDBE',
  humanInk: '#1F55B5',
  mixedInk: '#6B3FA6',
  aiInk: '#A8321F',
  bandWash: 'rgba(201, 184, 230, 0.78)',
  bandEdge: '#6E6178',
  markRing: '#FBF8F2',
  pendingGround: '#EAE3D6',
  pendingInk: '#5C4A1E',
  unreacted: '#E7E0D2',
};

const rgb = (hex: string): [number, number, number] => [
  parseInt(hex.slice(1, 3), 16),
  parseInt(hex.slice(3, 5), 16),
  parseInt(hex.slice(5, 7), 16),
];

const mix = (a: string, b: string, k: number): string => {
  const [r1, g1, b1] = rgb(a);
  const [r2, g2, b2] = rgb(b);
  const c = (x: number, y: number) => Math.round(x + (y - x) * k);
  return `rgb(${c(r1, r2)}, ${c(g1, g2)}, ${c(b1, b2)})`;
};

/** The strip's color at a position, 0 human through 0.5 mixed to 1 AI, as the site's linear gradient draws it. */
export const stripColor = (position: number): string =>
  position <= 0.5 ? mix(brand.human, brand.mixed, position / 0.5) : mix(brand.mixed, brand.ai, (position - 0.5) / 0.5);

export const originColor = (origin: string): string =>
  origin === 'human' ? brand.human : origin === 'mixed' ? brand.mixed : brand.ai;

export const originInk = (g: Ground, origin: string): string =>
  origin === 'human' ? g.humanInk : origin === 'mixed' ? g.mixedInk : g.aiInk;

// Single quotes inside, because these strings go into double-quoted style and SVG attributes.
export const fonts = {
  head: "'Source Serif 4 Variable', 'Source Serif 4', Georgia, serif",
  body: "'Inter Variable', Inter, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', Consolas, monospace",
};
