/**
 * GPU Detection and Benchmarking for Distributed Compute
 *
 * Detects WebGPU capabilities and runs performance benchmarks
 * to classify devices into compute tiers.
 */

// WebGPU type declarations for TypeScript
// These are available at runtime in browsers with WebGPU support
declare global {
  interface Navigator {
    gpu?: GPU;
  }

  interface GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  interface GPURequestAdapterOptions {
    powerPreference?: "low-power" | "high-performance";
  }

  interface GPUAdapter {
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
    requestAdapterInfo(): Promise<GPUAdapterInfo>;
    limits: GPUSupportedLimits;
    features: GPUSupportedFeatures;
  }

  interface GPUDeviceDescriptor {
    requiredFeatures?: Iterable<string>;
    requiredLimits?: Record<string, number>;
  }

  interface GPUDevice {
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createCommandEncoder(): GPUCommandEncoder;
    queue: GPUQueue;
    destroy(): void;
  }

  interface GPUShaderModuleDescriptor {
    code: string;
  }

  interface GPUShaderModule {}

  interface GPUComputePipelineDescriptor {
    layout: "auto" | GPUPipelineLayout;
    compute: GPUProgrammableStage;
  }

  interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
  }

  interface GPUPipelineLayout {}
  interface GPUComputePipeline {
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }
  interface GPUBindGroupLayout {}

  interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: GPUBindGroupEntry[];
  }

  interface GPUBindGroupEntry {
    binding: number;
    resource: GPUBufferBinding;
  }

  interface GPUBufferBinding {
    buffer: GPUBuffer;
  }

  interface GPUBindGroup {}

  interface GPUBufferDescriptor {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }

  interface GPUBuffer {
    mapAsync(mode: number): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  // WebGPU buffer usage flags
  var GPUBufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
  };

  // WebGPU map mode flags
  var GPUMapMode: {
    READ: number;
    WRITE: number;
  };

  interface GPUCommandEncoder {
    beginComputePass(): GPUComputePassEncoder;
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
    finish(): GPUCommandBuffer;
  }

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  }

  interface GPUCommandBuffer {}

  interface GPUQueue {
    submit(commandBuffers: GPUCommandBuffer[]): void;
    onSubmittedWorkDone(): Promise<void>;
    writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource): void;
  }

  interface GPUAdapterInfo {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  }

  interface GPUSupportedLimits {
    maxComputeWorkgroupSizeX: number;
    maxComputeWorkgroupSizeY: number;
    maxComputeWorkgroupSizeZ: number;
    maxComputeInvocationsPerWorkgroup: number;
  }

  interface GPUSupportedFeatures extends Set<string> {}
}

export interface GPUInfo {
  available: boolean;
  adapterInfo?: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  };
  limits?: {
    maxComputeWorkgroupSizeX: number;
    maxComputeWorkgroupSizeY: number;
    maxComputeWorkgroupSizeZ: number;
    maxComputeInvocationsPerWorkgroup: number;
  };
  features?: string[];
  estimatedMemoryMB?: number;
}

export interface BenchmarkResult {
  tflops: number;
  duration: number;
  operations: number;
  timestamp: number;
}

export type DeviceTier = 'power' | 'standard' | 'crowd';

export interface TierInfo {
  tier: DeviceTier;
  name: string;
  description: string;
  tokensPerHour: number;
  creditsPerHour: number;
  color: string;
  bgColor: string;
}

/**
 * Detect GPU capabilities using WebGPU API
 */
export async function detectGPU(): Promise<GPUInfo> {
  console.log('[GPU Detection] Starting...');
  console.log('[GPU Detection] navigator.gpu:', navigator.gpu);

  // Check if WebGPU is available
  if (!navigator.gpu) {
    console.log('[GPU Detection] WebGPU not available - navigator.gpu is undefined');
    return { available: false };
  }

  try {
    console.log('[GPU Detection] Requesting adapter...');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    console.log('[GPU Detection] Adapter:', adapter);

    if (!adapter) {
      console.log('[GPU Detection] No adapter returned');
      return { available: false };
    }

    // Get adapter info - handle both old and new API
    console.log('[GPU Detection] Getting adapter info...');
    let info: GPUAdapterInfo;

    if (typeof adapter.requestAdapterInfo === 'function') {
      // New API (Chrome 121+)
      info = await adapter.requestAdapterInfo();
    } else {
      // Fallback: try to access info directly or create placeholder
      // @ts-ignore - older Chrome versions may have different API
      info = (adapter as any).info || {
        vendor: 'Unknown',
        architecture: '',
        device: 'GPU',
        description: 'WebGPU Device',
      };
    }
    console.log('[GPU Detection] Adapter info:', info);

    // Get limits
    const limits = adapter.limits;

    // Get features
    const features = Array.from(adapter.features);

    // Estimate memory (very rough estimate based on device type)
    let estimatedMemoryMB = 2048; // Default 2GB
    const vendor = info.vendor.toLowerCase();
    const device = info.device.toLowerCase();

    if (device.includes('nvidia') || device.includes('rtx') || device.includes('gtx')) {
      estimatedMemoryMB = 8192; // Assume 8GB for NVIDIA GPUs
    } else if (device.includes('apple') || vendor.includes('apple')) {
      estimatedMemoryMB = 4096; // 4GB for Apple Silicon
    } else if (device.includes('amd') || device.includes('radeon')) {
      estimatedMemoryMB = 6144; // 6GB for AMD
    }

    return {
      available: true,
      adapterInfo: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      limits: {
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
      },
      features,
      estimatedMemoryMB,
    };
  } catch (error) {
    console.error('GPU detection error:', error);
    return { available: false };
  }
}

/**
 * Run a compute shader benchmark to estimate device performance
 * Runs matrix multiplication to measure TFLOPS
 */
export async function runBenchmark(): Promise<BenchmarkResult> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not available');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No GPU adapter found');
  }

  const device = await adapter.requestDevice();

  // Matrix multiplication shader (simplified for benchmarking)
  const shaderCode = `
    @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
    @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
    @group(0) @binding(2) var<storage, read_write> result: array<f32>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let size = 256u;
      let row = global_id.x;
      let col = global_id.y;

      if (row >= size || col >= size) {
        return;
      }

      var sum = 0.0;
      for (var i = 0u; i < size; i = i + 1u) {
        sum = sum + matrixA[row * size + i] * matrixB[i * size + col];
      }

      result[row * size + col] = sum;
    }
  `;

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  // Create pipeline
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  // Matrix size (256x256 for quick benchmark)
  const matrixSize = 256;
  const numElements = matrixSize * matrixSize;
  const bufferSize = numElements * Float32Array.BYTES_PER_ELEMENT;

  // Create buffers
  const bufferA = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bufferB = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const resultBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Initialize with random data
  const dataA = new Float32Array(numElements).map(() => Math.random());
  const dataB = new Float32Array(numElements).map(() => Math.random());

  device.queue.writeBuffer(bufferA, 0, dataA);
  device.queue.writeBuffer(bufferB, 0, dataB);

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufferA } },
      { binding: 1, resource: { buffer: bufferB } },
      { binding: 2, resource: { buffer: resultBuffer } },
    ],
  });

  // Run benchmark
  const startTime = performance.now();

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(
    Math.ceil(matrixSize / 8),
    Math.ceil(matrixSize / 8)
  );
  passEncoder.end();

  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const duration = performance.now() - startTime;

  // Calculate TFLOPS
  // Matrix multiplication: N^3 * 2 operations (multiply and add)
  const operations = Math.pow(matrixSize, 3) * 2;
  const flops = operations / (duration / 1000); // FLOPS
  const tflops = flops / 1e12; // Convert to TFLOPS

  // Cleanup
  bufferA.destroy();
  bufferB.destroy();
  resultBuffer.destroy();
  device.destroy();

  return {
    tflops,
    duration,
    operations,
    timestamp: Date.now(),
  };
}

/**
 * Classify device tier based on benchmark results
 */
export function classifyTier(benchmark: BenchmarkResult): DeviceTier {
  const { tflops } = benchmark;

  // Tier classification based on TFLOPS
  if (tflops >= 5.0) {
    return 'power'; // High-end GPUs (RTX 3080+, Apple M2 Max+)
  } else if (tflops >= 1.0) {
    return 'standard'; // Mid-range GPUs (GTX 1660, M1/M2, integrated)
  } else {
    return 'crowd'; // Lower-end devices (older GPUs, mobile)
  }
}

/**
 * Get tier information including contribution estimates
 */
export function getTierInfo(tier: DeviceTier): TierInfo {
  const tierMap: Record<DeviceTier, TierInfo> = {
    power: {
      tier: 'power',
      name: 'Power Tier',
      description: 'High-end GPU - Perfect for training and inference',
      tokensPerHour: 50000,
      creditsPerHour: 100,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
    },
    standard: {
      tier: 'standard',
      name: 'Standard Tier',
      description: 'Mid-range GPU - Great for inference and small models',
      tokensPerHour: 15000,
      creditsPerHour: 30,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    crowd: {
      tier: 'crowd',
      name: 'Crowd Tier',
      description: 'Entry-level GPU - Contributes to distributed inference',
      tokensPerHour: 5000,
      creditsPerHour: 10,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
    },
  };

  return tierMap[tier];
}

/**
 * Format TFLOPS for display
 */
export function formatTFLOPS(tflops: number): string {
  if (tflops >= 1) {
    return `${tflops.toFixed(2)} TFLOPS`;
  } else {
    const gflops = tflops * 1000;
    return `${gflops.toFixed(0)} GFLOPS`;
  }
}

/**
 * Format GPU memory for display
 */
export function formatMemory(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb} MB`;
}
