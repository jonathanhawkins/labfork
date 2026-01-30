import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { mkdir } from "fs/promises";
import { join } from "path";
import { randomUUID } from "crypto";
import { existsSync } from "fs";

const INFERENCE_SCRIPT = "inference/generate_with_chatterbox.py";
const OUTPUT_DIR = "public/generated";

interface GenerateRequest {
  text: string | number;
}

export async function POST(request: NextRequest) {
  try {
    const params: GenerateRequest = await request.json();

    if (!params.text || typeof params.text !== "string") {
      return NextResponse.json({ error: "Text is required" }, { status: 400 });
    }

    const outputId = randomUUID();
    const outputFilename = `chatterbox_${outputId}.wav`;
    const outputPath = join(process.cwd(), OUTPUT_DIR, outputFilename);

    const outputDir = join(process.cwd(), OUTPUT_DIR);
    if (!existsSync(outputDir)) {
      await mkdir(outputDir, { recursive: true });
    }

    const scriptPath = join(process.cwd(), "..", INFERENCE_SCRIPT);

    if (!existsSync(scriptPath)) {
      // Demo mode - return mock response
      return NextResponse.json({
        audioUrl: "/demo-audio-placeholder.wav",
        message: "Demo mode: inference script not found at " + scriptPath,
      });
    }

    const result = await runInference(scriptPath, params, outputPath);

    if (!result.success) {
      return NextResponse.json({ error: result.error }, { status: 500 });
    }

    return NextResponse.json({
      audioUrl: `/generated/${outputFilename}`,
      prosodyData: result.prosodyData,
    });

  } catch (error) {
    console.error("chatterbox API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

async function runInference(
  scriptPath: string,
  params: GenerateRequest,
  outputPath: string
): Promise<{ success: boolean; error?: string; prosodyData?: object }> {
  return new Promise((resolve) => {
    const args = [
      scriptPath,
      "--text", params.text.toString(),
      "--output", outputPath,
    ];

    const proc = spawn("python3", args, {
      cwd: join(process.cwd(), ".."),
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => { stdout += data.toString(); });
    proc.stderr.on("data", (data) => { stderr += data.toString(); });

    proc.on("close", (code) => {
      if (code !== 0) {
        resolve({ success: false, error: stderr || `Exit code ${code}` });
        return;
      }
      resolve({ success: true, prosodyData: {} });
    });

    proc.on("error", (err) => {
      resolve({ success: false, error: err.message });
    });

    setTimeout(() => {
      proc.kill();
      resolve({ success: false, error: "Timeout after 60s" });
    }, 60000);
  });
}
