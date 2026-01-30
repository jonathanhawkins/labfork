import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

const RUNS_DIR = path.join(process.cwd(), '..', 'outputs', 'research', 'runs')

const TRAINING_CONFIG_DIR = path.join(process.cwd(), '..', 'training', 'config')

// Base/hardware configs to exclude (not research techniques)
const EXCLUDED_CONFIGS = new Set([
  'baseline_no_prosody',
  'm4_pro',
  'm4_pro_deepseek',
  'prosody_conditioned',
  'prosody_joint_training',
  'prosody_joint_v4',
  'prosody_v5',
  'prosody_v7_balanced',
  'rtx_4090',
  'rtx_4090_deepseek',
  'rtx_4090_lora',
  'rtx_4090_lora_450',
])

// Format config name to readable technique name
function formatTechniqueName(configName: string): string {
  return configName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

async function extractTechniquesFromConfigs(): Promise<string[]> {
  try {
    const files = await fs.readdir(TRAINING_CONFIG_DIR)
    const techniques: string[] = []

    for (const file of files) {
      if (file.endsWith('.yaml')) {
        const name = file.replace('.yaml', '')
        if (!EXCLUDED_CONFIGS.has(name)) {
          techniques.push(formatTechniqueName(name))
        }
      }
    }

    return techniques.sort()
  } catch (e) {
    console.error('Failed to read training configs:', e)
    return []
  }
}

async function loadRuns() {
  try {
    const entries = await fs.readdir(RUNS_DIR, { withFileTypes: true })
    const runs = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory())
        .map(async (entry) => {
          const runFile = path.join(RUNS_DIR, entry.name, 'run.json')
          try {
            const content = await fs.readFile(runFile, 'utf-8')
            return JSON.parse(content)
          } catch {
            return null
          }
        })
    )
    return runs.filter(Boolean).sort((a, b) => {
      const aTime = a?.created_at || ''
      const bTime = b?.created_at || ''
      return bTime.localeCompare(aTime)
    })
  } catch (e) {
    return []
  }
}

function runToDecision(run: any) {
  return {
    name: run.title || run.technique || run.run_id,
    reason: run.review?.reason || '',
    date: run.review?.reviewed_at || run.updated_at || run.created_at,
    impact: run.metrics ? 'metrics' : '',
    gate: run.review?.status || 'pending',
  }
}

export async function GET() {
  const runs = await loadRuns()
  const techniques = await extractTechniquesFromConfigs()

  const approvedRuns = runs.filter((r: any) => r.review?.status === 'approved')
  const rejectedRuns = runs.filter((r: any) => r.review?.status === 'rejected')
  const pendingRuns = runs.filter(
    (r: any) => !r.review?.status || r.review?.status === 'pending'
  )
  const evaluatedRuns = runs.filter((r: any) => r.metrics && Object.keys(r.metrics).length > 0)

  const stats = {
    total: runs.length,
    approved: approvedRuns.length,
    rejected: rejectedRuns.length,
    evaluated: evaluatedRuns.length,
    pending: pendingRuns.length,
  }

  return NextResponse.json({
    stats,
    approved: approvedRuns.map(runToDecision),
    rejected: rejectedRuns.map(runToDecision),
    pending: pendingRuns.map(runToDecision),
    evaluated: evaluatedRuns.map(runToDecision),
    techniques,
    runs,
    latestRun: runs[0] || null,
  })
}
