/**
 * Domain-specific 3D Props
 *
 * Custom 3D components for each research domain.
 */

// Quant Trading
export {
  createTradingTerminal3D,
  animateTradingTerminal3D,
  disposeTradingTerminal3D,
} from "./TradingTerminal3D";
export type { TradingTerminal3DRefs, TradingTerminal3DOptions } from "./TradingTerminal3D";

// Game AI
export {
  createGameController3D,
  animateGameController3D,
  disposeGameController3D,
} from "./GameController3D";
export type { GameController3DRefs, GameController3DOptions } from "./GameController3D";

// Robotics
export {
  createRobotArm3D,
  animateRobotArm3D,
  disposeRobotArm3D,
} from "./RobotArm3D";
export type { RobotArm3DRefs, RobotArm3DOptions } from "./RobotArm3D";

// Drug Discovery
export {
  createMoleculeViewer3D,
  animateMoleculeViewer3D,
  disposeMoleculeViewer3D,
} from "./MoleculeViewer3D";
export type { MoleculeViewer3DRefs, MoleculeViewer3DOptions } from "./MoleculeViewer3D";

// Climate Modeling
export {
  createEarthGlobe3D,
  animateEarthGlobe3D,
  disposeEarthGlobe3D,
} from "./EarthGlobe3D";
export type { EarthGlobe3DRefs, EarthGlobe3DOptions } from "./EarthGlobe3D";

// NLP Research
export {
  createTextCorpus3D,
  animateTextCorpus3D,
  disposeTextCorpus3D,
} from "./TextCorpus3D";
export type { TextCorpus3DRefs, TextCorpus3DOptions } from "./TextCorpus3D";

// Computer Vision
export {
  createImageGrid3D,
  animateImageGrid3D,
  disposeImageGrid3D,
} from "./ImageGrid3D";
export type { ImageGrid3DRefs, ImageGrid3DOptions } from "./ImageGrid3D";
