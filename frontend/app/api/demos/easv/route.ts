import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { writeFile, mkdir, unlink } from "fs/promises";
import { join } from "path";
import { randomUUID } from "crypto";
import { existsSync } from "fs";

// Path to Python inference script (relative to project root)
const INFERENCE_SCRIPT = "inference/generate_with_easv.py";

// Output directory for generated audio
const OUTPUT_DIR = "public/generated";

interface GenerateRequest {
  text: string;
  emotion: string;
  intensity: number;
}

export async function POST(request: NextRequest) {
  try {
    const body: GenerateRequest = await request.json();
    const { text, emotion, intensity } = body;

    // Validate input
    if (!text || typeof text !== "string") {
      return NextResponse.json(
        { error: "Text is required" },
        { status: 400 }
      );
    }

    if (!emotion || typeof emotion !== "string") {
      return NextResponse.json(
        { error: "Emotion is required" },
        { status: 400 }
      );
    }

    if (typeof intensity !== "number" || intensity < 0 || intensity > 1) {
      return NextResponse.json(
        { error: "Intensity must be between 0 and 1" },
        { status: 400 }
      );
    }

    // Generate unique output filename
    const outputId = randomUUID();
    const outputFilename = `easv_${outputId}.wav`;
    const outputPath = join(process.cwd(), OUTPUT_DIR, outputFilename);

    // Ensure output directory exists
    const outputDir = join(process.cwd(), OUTPUT_DIR);
    if (!existsSync(outputDir)) {
      await mkdir(outputDir, { recursive: true });
    }

    // Call Python inference script
    const scriptPath = join(process.cwd(), "..", INFERENCE_SCRIPT);

    // Check if the script exists
    if (!existsSync(scriptPath)) {
      console.error(`Inference script not found: ${scriptPath}`);

      // For demo purposes, return a mock response if script doesn't exist
      // In production, this would be an error
      return NextResponse.json({
        audioUrl: "/demo-audio-placeholder.wav",
        prosodyData: {
          emotion,
          intensity,
          f0Mean: 180 + intensity * 50,
          energyMean: 65 + intensity * 10,
        },
        message: "Demo mode: inference script not found",
      });
    }

    // Run the inference script
    const result = await runPythonInference(scriptPath, {
      text,
      emotion,
      intensity,
      output: outputPath,
    });

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || "Inference failed" },
        { status: 500 }
      );
    }

    // Return the audio URL
    return NextResponse.json({
      audioUrl: `/generated/${outputFilename}`,
      prosodyData: result.prosodyData,
    });

  } catch (error) {
    console.error("EASV API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

interface InferenceResult {
  success: boolean;
  error?: string;
  prosodyData?: {
    emotion: string;
    intensity: number;
    f0Mean: number;
    energyMean: number;
  };
}

async function runPythonInference(
  scriptPath: string,
  params: {
    text: string;
    emotion: string;
    intensity: number;
    output: string;
  }
): Promise<InferenceResult> {
  return new Promise((resolve) => {
    const args = [
      scriptPath,
      "--text", params.text,
      "--emotion", params.emotion,
      "--intensity", params.intensity.toString(),
      "--output", params.output,
      "--json-output", // Request JSON prosody data
    ];

    const proc = spawn("python3", args, {
      cwd: join(process.cwd(), ".."),
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        console.error("Inference stderr:", stderr);
        resolve({
          success: false,
          error: `Inference failed with code ${code}: ${stderr}`,
        });
        return;
      }

      // Try to parse prosody data from stdout
      let prosodyData;
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          prosodyData = JSON.parse(jsonMatch[0]);
        }
      } catch {
        // Prosody data is optional
      }

      resolve({
        success: true,
        prosodyData: prosodyData || {
          emotion: params.emotion,
          intensity: params.intensity,
          f0Mean: 180,
          energyMean: 70,
        },
      });
    });

    proc.on("error", (err) => {
      resolve({
        success: false,
        error: `Failed to spawn Python: ${err.message}`,
      });
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      proc.kill();
      resolve({
        success: false,
        error: "Inference timed out after 60 seconds",
      });
    }, 60000);
  });
}
