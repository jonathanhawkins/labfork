'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CheckCircle2, XCircle, Clock, AlertTriangle, Target, Beaker, ArrowRight } from "lucide-react"

interface ResearchStats {
  total: number
  approved: number
  rejected: number
  evaluated: number
  pending: number
}

interface Decision {
  name: string
  reason?: string
  gate?: string
  date?: string
  impact?: string
}

interface ResearchData {
  stats: ResearchStats
  approved: Decision[]
  rejected: Decision[]
  pending: Decision[]
  evaluated: string[]
  runs?: RunRecord[]
  latestRun?: RunRecord | null
}

interface RunRecord {
  run_id: string
  title?: string
  technique?: string
  created_at?: string
  updated_at?: string
  status?: string
  metrics?: Record<string, any>
  review?: {
    status?: string
    reviewer?: string
    reason?: string
    reviewed_at?: string
  }
}

export default function ResearchPage() {
  const [data, setData] = useState<ResearchData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch('/api/research')
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const json = await res.json()
        setData(json)
        setError(null)
      } catch (e) {
        console.error('Failed to fetch research data:', e)
        setError(e instanceof Error ? e.message : 'Failed to load research data')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const coreGoal = {
    question: "Can explicit multi-layer prosody labels improve voice cloning when training data is limited?",
    metrics: [
      { name: "F0 Separation", target: "Happy > Sad by 30+ Hz", status: "in_progress" },
      { name: "Emotion Accuracy", target: "≥ 50%", status: "partial" },
      { name: "Speaker Similarity", target: "≥ 0.7", status: "unknown" },
    ],
    currentResults: {
      f0Correlation: 0.328,
      emotionAccuracy: "50% (2/4)",
      v7Status: "Needs retraining with LoRA fix"
    }
  }

  const latestRun = data?.latestRun
  const latestMetrics = latestRun?.metrics
  const f0Label = latestMetrics?.f0_separation_hz !== undefined ? "F0 Separation (Hz)" : "F0 Correlation"
  const f0Value = latestMetrics?.f0_separation_hz ?? coreGoal.currentResults.f0Correlation
  const emotionValue = latestMetrics?.emotion_accuracy !== undefined
    ? `${(latestMetrics.emotion_accuracy * 100).toFixed(1)}%`
    : coreGoal.currentResults.emotionAccuracy
  const statusValue = latestRun?.status || coreGoal.currentResults.v7Status
  const currentLabel = latestRun
    ? (latestRun.title || latestRun.technique || latestRun.run_id)
    : "v3"

  const weeklyFocus = [
    { task: "Complete V7 verification", priority: "P0", status: "blocked" },
    { task: "Run quick_eval.py on V7 checkpoint", priority: "P0", status: "pending" },
    { task: "Evaluate EmoKnob (direction vectors)", priority: "P1", status: "pending" },
    { task: "Evaluate Emo-FiLM (word-level emotion)", priority: "P1", status: "pending" },
    { task: "Evaluate Activation Steering", priority: "P1", status: "pending" },
  ]

  const doNot = [
    "Start new research tasks",
    "Add to CLAUDE.md",
    "Implement unevaluated techniques",
  ]

  if (loading) {
    return (
      <div className="min-h-screen bg-background p-8 flex items-center justify-center">
        <div className="text-muted-foreground">Loading research data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background p-8 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-foreground mb-2">Unable to load research data</h2>
          <p className="text-muted-foreground mb-4">{error}</p>
          <Button onClick={() => window.location.reload()}>Retry</Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Research Dashboard</h1>
            <p className="text-muted-foreground mt-1">Decision support for research management</p>
          </div>
          <Badge variant="outline" className="text-lg px-4 py-2">
            {data?.stats.pending || 0} pending evaluation
          </Badge>
        </div>

        {/* Core Goal */}
        <Card className="border-primary/50 bg-primary/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              Core Research Goal
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-lg font-medium mb-4">{coreGoal.question}</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              {coreGoal.metrics.map((metric, i) => (
                <div key={i} className="p-4 rounded-lg bg-background border">
                  <div className="flex items-center gap-2 mb-2">
                    {metric.status === 'complete' && <CheckCircle2 className="h-4 w-4 text-green-500" />}
                    {metric.status === 'in_progress' && <Clock className="h-4 w-4 text-yellow-500" />}
                    {metric.status === 'partial' && <AlertTriangle className="h-4 w-4 text-orange-500" />}
                    {metric.status === 'unknown' && <Beaker className="h-4 w-4 text-muted-foreground" />}
                    <span className="font-medium">{metric.name}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">{metric.target}</p>
                </div>
              ))}
            </div>

            <div className="p-4 rounded-lg bg-background border">
              <h4 className="font-medium mb-2">Current Results ({currentLabel})</h4>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">{f0Label}:</span>
                  <span className="ml-2 font-mono">{f0Value}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Emotion Accuracy:</span>
                  <span className="ml-2 font-mono">{emotionValue}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">V7 Status:</span>
                  <span className="ml-2 text-yellow-500">{statusValue}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-4xl font-bold">{data?.stats.total || 0}</div>
              <p className="text-muted-foreground text-sm">Total Techniques</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-4xl font-bold text-green-500">{data?.stats.approved || 0}</div>
              <p className="text-muted-foreground text-sm">Approved</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-4xl font-bold text-red-500">{data?.stats.rejected || 0}</div>
              <p className="text-muted-foreground text-sm">Rejected</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-4xl font-bold text-blue-500">{data?.stats.evaluated || 0}</div>
              <p className="text-muted-foreground text-sm">Evaluated</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-4xl font-bold text-yellow-500">{data?.stats.pending || 0}</div>
              <p className="text-muted-foreground text-sm">Pending</p>
            </CardContent>
          </Card>
        </div>

        {/* Latest Runs */}
        {data?.runs && data.runs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Latest Runs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {data.runs.slice(0, 5).map((run, i) => (
                  <div key={run.run_id || i} className="flex items-center gap-3 p-3 rounded-lg bg-muted/40">
                    <span className="flex-1 font-medium">
                      {run.title || run.technique || run.run_id}
                    </span>
                    <Badge variant="outline">{run.review?.status || "pending"}</Badge>
                    <span className="text-xs text-muted-foreground">
                      {run.created_at || run.updated_at || ""}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Weekly Focus */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ArrowRight className="h-5 w-5" />
                Weekly Focus
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {weeklyFocus.map((item, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                    <Badge variant={item.priority === 'P0' ? 'destructive' : 'secondary'}>
                      {item.priority}
                    </Badge>
                    <span className="flex-1">{item.task}</span>
                    <Badge variant="outline">{item.status}</Badge>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 rounded-lg bg-red-500/10 border border-red-500/30">
                <h4 className="font-medium text-red-500 mb-2 flex items-center gap-2">
                  <XCircle className="h-4 w-4" />
                  DO NOT
                </h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  {doNot.map((item, i) => (
                    <li key={i}>• {item}</li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 rounded-lg border bg-muted/30">
                <h4 className="font-medium mb-2">Run Quick Evaluation</h4>
                <code className="text-sm bg-background px-2 py-1 rounded block mb-3">
                  python inference/quick_eval.py --checkpoint &lt;model&gt;
                </code>
                <p className="text-sm text-muted-foreground">
                  30-minute end-to-end evaluation for any technique
                </p>
              </div>

              <div className="p-4 rounded-lg border bg-muted/30">
                <h4 className="font-medium mb-2">Research Lead Commands</h4>
                <div className="space-y-2 text-sm font-mono">
                  <div><code>.skills/research-manager/rm lead status</code></div>
                  <div><code>.skills/research-manager/rm lead evaluate</code></div>
                  <div><code>.skills/research-manager/rm lead focus</code></div>
                  <div><code>.skills/research-manager/rm lead decide "technique"</code></div>
                </div>
              </div>

              <div className="p-4 rounded-lg border bg-muted/30">
                <h4 className="font-medium mb-2">Decision Framework</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Gate 1:</span>
                    <span className="text-muted-foreground">Relevance (helps prosody?)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Gate 2:</span>
                    <span className="text-muted-foreground">Testable in 4 hours?</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Gate 3:</span>
                    <span className="text-muted-foreground">Expected impact?</span>
                  </div>
                </div>
              </div>

              <Button className="w-full" variant="outline">
                View Decision Framework
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Decisions Log */}
        {(data?.approved?.length || data?.rejected?.length) && (
          <Card>
            <CardHeader>
              <CardTitle>Recent Decisions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {data?.approved?.slice(0, 5).map((d, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-green-500/10">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span className="flex-1">{typeof d === 'string' ? d : d.name}</span>
                    <Badge variant="outline" className="text-green-500">Approved</Badge>
                  </div>
                ))}
                {data?.rejected?.slice(0, 5).map((d, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-red-500/10">
                    <XCircle className="h-4 w-4 text-red-500" />
                    <span className="flex-1">{typeof d === 'string' ? d : d.name}</span>
                    <span className="text-sm text-muted-foreground">{typeof d === 'object' ? d.reason : ''}</span>
                    <Badge variant="outline" className="text-red-500">Rejected</Badge>
                  </div>
                ))}
                {!data?.approved?.length && !data?.rejected?.length && (
                  <p className="text-muted-foreground text-center py-4">
                    No decisions recorded yet. Use the lead commands to evaluate techniques.
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
