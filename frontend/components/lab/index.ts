// Lab 3D Visualization System
// Modular components for building cute Katamari-style 3D scenes

// Types
export * from "./types";

// Colors
export * from "./colors";

// Activity System (new)
export * from "./activities";

// Props System (new)
export * from "./props";

// Public Components
export { PublicLabView } from "./PublicLabView";
export { SuggestionForm } from "./SuggestionForm";
export { SuggestionList } from "./SuggestionList";

// 3D Components
export {
  createAgent3D,
  animateAgent3D,
  updateAgentStatus,
} from "./Agent3D";

export {
  createWorkstation3D,
  animateWorkstationScreen,
  type WorkstationRefs,
} from "./Workstation3D";

export {
  createDataParticles3D,
  animateDataParticles,
  createParticleStream,
  type ParticleSystemRefs,
} from "./DataParticles3D";

export {
  createDataHub3D,
  animateDataHub,
  createHubConnections,
  type DataHubRefs,
} from "./DataHub3D";

export {
  createPlant3D,
  createFloatingCube3D,
  animateFloatingCube,
  createGroundPattern3D,
  createAllDecorations,
} from "./Decorations3D";
