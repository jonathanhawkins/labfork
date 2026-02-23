// Simulation Types for LabFork Research Labs

export type SimulationType =
  | 'water_harvester'
  | 'droplet_dynamics'
  | 'heat_transfer'
  | 'solar_concentration'
  | 'sorbent_desorption';

export type SimulationMode = 'quick' | 'full';

export type SimulationStatus = 'pending' | 'running' | 'completed' | 'failed';

// Water Harvester specific parameters
export interface WaterHarvesterParams {
  sorbent_width_cm: number;
  sorbent_depth_cm: number;
  sorbent_thickness_cm: number;
  mirror_count: number;
  mirror_angle: number;
  humidity_percent: number;
  temperature_ambient_c: number;
  surface_pattern: 'beetle' | 'flat' | 'trichome' | 'channel';
  sorbent_type: 'cacl2_silica' | 'mof_808' | 'cal_gel';
}

// Generic simulation parameters (extensible per lab type)
export interface SimulationParams {
  type: SimulationType;
  parameters: Partial<WaterHarvesterParams> & Record<string, unknown>;
  mode: SimulationMode;
}

// Water harvester results
export interface WaterHarvesterResults {
  collection_rate_ml_per_hour: number;
  daily_yield_liters: number;
  efficiency_percent: number;
  peak_temperature_c: number;
  condensation_rate_g_per_m2_hour: number;
  sorbent_saturation_percent?: number;
  cycle_time_hours?: number;
}

// Generic simulation result
export interface SimulationResult {
  id: string;
  status: SimulationStatus;
  params: SimulationParams;
  results?: WaterHarvesterResults | Record<string, number>;
  created_at: string;
  completed_at?: string;
  error?: string;
  device_id?: string; // Which device ran it
  compute_time_ms?: number;
}

// API response types
export interface SimulationResponse {
  success: boolean;
  simulation?: SimulationResult;
  error?: string;
  message?: string;
}

export interface SimulationListResponse {
  success: boolean;
  simulations: SimulationResult[];
  total: number;
  error?: string;
}

// Client-side simulation hook state
export interface UseSimulationState {
  simulation: SimulationResult | null;
  isLoading: boolean;
  isPolling: boolean;
  error: string | null;
  runSimulation: (params: SimulationParams) => Promise<void>;
  cancelSimulation: () => void;
}

// Research paper reference for results
export interface ResearchReference {
  title: string;
  authors: string;
  year: number;
  url: string;
  relevantFinding: string;
}

// Comparison data for validation
export interface ValidationComparison {
  source: 'research' | 'community' | 'simulation';
  reference?: ResearchReference;
  value: number;
  unit: string;
  conditions: string;
}
