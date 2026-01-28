// Katamari Damacy style pastel color palette for Lab 3D

export const LAB_COLORS = {
  // Scene
  sky: 0xffeef5,
  ground: 0xb8e6c1,
  groundAccent: 0x9dd4a8,
  fog: 0xffeef5,

  // Agents
  codex: 0xffb3ba,    // Soft pink
  opus: 0xbae1ff,     // Soft blue
  explorer: 0xffffba, // Soft yellow
  planner: 0xbaffc9,  // Soft green

  // Furniture
  desk: 0xffe4b8,     // Soft orange/wood
  deskLeg: 0xccb088,
  monitor: 0x333333,
  keyboard: 0x444444,

  // Screens
  screen: 0x2a2a3a,
  screenGlow: 0x66ffaa,

  // Effects
  particles: 0xffccee,
  hub: 0xffaacc,

  // Decorations
  pot: 0xcc8866,
  plant: 0x66bb66,

  // UI
  eyes: 0x2a2a2a,
  eyeHighlight: 0xffffff,
  antenna: 0x666666,
  antennaWorking: 0x44ff44,
  antennaThinking: 0xffff44,
} as const;

export type LabColorKey = keyof typeof LAB_COLORS;

// Get a rainbow color for decorative elements
export function getRainbowColor(index: number, total: number): number {
  const hue = index / total;
  const color = new (require("three").Color)();
  color.setHSL(hue, 0.6, 0.7);
  return color.getHex();
}
