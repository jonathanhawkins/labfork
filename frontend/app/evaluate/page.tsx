"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Check,
  Download,
  Upload,
  BarChart3,
  Volume2,
  Circle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface TestPair {
  id: string;
  text: string;
  emotion: string;
  audioA: string;
  audioB: string;
  modelA: "baseline" | "prosody";
  modelB: "baseline" | "prosody";
}

interface TestResult {
  pairId: string;
  selectedAudio: "A" | "B" | "tie";
  selectedModel: "baseline" | "prosody" | "tie";
  timestamp: string;
  question: string;
}

const sampleTestPairs: TestPair[] = [
  {
    id: "test_001",
    text: "I just received some news about the project.",
    emotion: "neutral",
    audioA: "/audio/baseline_neutral_001.wav",
    audioB: "/audio/prosody_neutral_001.wav",
    modelA: "baseline",
    modelB: "prosody",
  },
  {
    id: "test_002",
    text: "This is absolutely incredible news!",
    emotion: "happy",
    audioA: "/audio/prosody_happy_001.wav",
    audioB: "/audio/baseline_happy_001.wav",
    modelA: "prosody",
    modelB: "baseline",
  },
  {
    id: "test_003",
    text: "I can't believe they would do this.",
    emotion: "sad",
    audioA: "/audio/baseline_sad_001.wav",
    audioB: "/audio/prosody_sad_001.wav",
    modelA: "baseline",
    modelB: "prosody",
  },
];

const questions = [
  { id: "natural", text: "Which sounds more natural?" },
  { id: "emotion", text: "Which better expresses the intended emotion?" },
  { id: "voice", text: "Which sounds more like a real human voice?" },
];

// Collapsible Section
function Section({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-foreground-bright hover:text-foreground transition-colors"
      >
        <span className="text-sm">{title}</span>
        <span className="text-muted-foreground">{isOpen ? "-" : "+"}</span>
      </button>
      {isOpen && <div className="px-4 pb-4 animate-fade-in">{children}</div>}
    </div>
  );
}

// Stat Row
function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

export default function EvaluatePage() {
  const [testPairs, setTestPairs] = useState<TestPair[]>([]);
  const [currentPairIndex, setCurrentPairIndex] = useState(0);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [results, setResults] = useState<TestResult[]>([]);
  const [isPlaying, setIsPlaying] = useState<"A" | "B" | null>(null);
  const [showResults, setShowResults] = useState(false);

  const audioRefA = useRef<HTMLAudioElement>(null);
  const audioRefB = useRef<HTMLAudioElement>(null);

  const currentPair = testPairs[currentPairIndex];
  const currentQuestion = questions[currentQuestionIndex];
  const progress =
    testPairs.length > 0
      ? ((currentPairIndex * questions.length + currentQuestionIndex) /
          (testPairs.length * questions.length)) *
        100
      : 0;

  useEffect(() => {
    const shuffled = [...sampleTestPairs].sort(() => Math.random() - 0.5);
    const randomized = shuffled.map((pair) => {
      if (Math.random() > 0.5) {
        return {
          ...pair,
          audioA: pair.audioB,
          audioB: pair.audioA,
          modelA: pair.modelB,
          modelB: pair.modelA,
        };
      }
      return pair;
    });
    setTestPairs(randomized);
  }, []);

  const playAudio = (which: "A" | "B") => {
    audioRefA.current?.pause();
    audioRefB.current?.pause();

    if (audioRefA.current) audioRefA.current.currentTime = 0;
    if (audioRefB.current) audioRefB.current.currentTime = 0;

    if (which === "A") {
      audioRefA.current?.play();
    } else {
      audioRefB.current?.play();
    }
    setIsPlaying(which);
  };

  const stopAudio = () => {
    audioRefA.current?.pause();
    audioRefB.current?.pause();
    setIsPlaying(null);
  };

  const handleAudioEnd = () => {
    setIsPlaying(null);
  };

  const selectWinner = (selection: "A" | "B" | "tie") => {
    if (!currentPair) return;

    const result: TestResult = {
      pairId: currentPair.id,
      selectedAudio: selection,
      selectedModel:
        selection === "tie"
          ? "tie"
          : selection === "A"
          ? currentPair.modelA
          : currentPair.modelB,
      timestamp: new Date().toISOString(),
      question: currentQuestion.id,
    };

    setResults((prev) => [...prev, result]);

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex((prev) => prev + 1);
    } else if (currentPairIndex < testPairs.length - 1) {
      setCurrentPairIndex((prev) => prev + 1);
      setCurrentQuestionIndex(0);
    } else {
      setShowResults(true);
    }

    stopAudio();
  };

  const calculateStats = () => {
    const stats = {
      total: results.length,
      baseline: 0,
      prosody: 0,
      tie: 0,
      byQuestion: {} as Record<string, { baseline: number; prosody: number; tie: number }>,
    };

    for (const result of results) {
      if (result.selectedModel === "baseline") stats.baseline++;
      else if (result.selectedModel === "prosody") stats.prosody++;
      else stats.tie++;

      if (!stats.byQuestion[result.question]) {
        stats.byQuestion[result.question] = { baseline: 0, prosody: 0, tie: 0 };
      }
      if (result.selectedModel === "baseline")
        stats.byQuestion[result.question].baseline++;
      else if (result.selectedModel === "prosody")
        stats.byQuestion[result.question].prosody++;
      else stats.byQuestion[result.question].tie++;
    }

    return stats;
  };

  const exportResults = () => {
    const stats = calculateStats();
    const exportData = {
      timestamp: new Date().toISOString(),
      summary: stats,
      individual_results: results,
      conclusion:
        stats.prosody > stats.baseline
          ? `Prosody-conditioned model preferred ${stats.prosody}/${stats.total} times`
          : stats.baseline > stats.prosody
          ? `Baseline model preferred ${stats.baseline}/${stats.total} times`
          : "No clear preference between models",
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ab_test_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    toast.success("Results exported!");
  };

  const resetTest = () => {
    setResults([]);
    setCurrentPairIndex(0);
    setCurrentQuestionIndex(0);
    setShowResults(false);
    const shuffled = [...sampleTestPairs].sort(() => Math.random() - 0.5);
    setTestPairs(shuffled);
  };

  // Results View
  if (showResults) {
    const stats = calculateStats();
    const prosodyWinRate =
      stats.total > 0 ? (stats.prosody / stats.total) * 100 : 0;
    const baselineWinRate =
      stats.total > 0 ? (stats.baseline / stats.total) * 100 : 0;

    return (
      <div className="min-h-screen bg-background flex">
        {/* Left Sidebar */}
        <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
          <Section title="Summary" defaultOpen>
            <div className="space-y-2">
              <StatRow label="Total Tests" value={stats.total} />
              <StatRow label="Baseline Wins" value={stats.baseline} />
              <StatRow label="Prosody Wins" value={stats.prosody} />
              <StatRow label="Ties" value={stats.tie} />
            </div>
          </Section>

          <Section title="By Question" defaultOpen>
            <div className="space-y-3">
              {Object.entries(stats.byQuestion).map(([q, scores]) => (
                <div key={q} className="space-y-1">
                  <span className="text-xs text-muted-foreground capitalize">{q}</span>
                  <div className="flex gap-2 text-xs">
                    <span className="text-foreground">B: {scores.baseline}</span>
                    <span className="text-foreground-bright">P: {scores.prosody}</span>
                    <span className="text-foreground-subtle">T: {scores.tie}</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Actions">
            <div className="space-y-2">
              <Button onClick={exportResults} className="w-full">
                <Download className="w-4 h-4 mr-2" />
                Export Results
              </Button>
              <Button onClick={resetTest} variant="outline" className="w-full">
                <RotateCcw className="w-4 h-4 mr-2" />
                New Test
              </Button>
            </div>
          </Section>
        </aside>

        {/* Main Content */}
        <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
          <div className="p-8 max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <BarChart3 className="w-8 h-8 text-foreground-bright mx-auto mb-4" />
              <h1 className="text-lg text-foreground-bright mb-2">A/B Test Results</h1>
              <p className="text-sm text-muted-foreground">
                {results.length} comparisons completed
              </p>
            </div>

            {/* Overall Stats */}
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-background-card border border-border rounded p-4 text-center">
                <p className="text-2xl text-foreground-bright">{stats.baseline}</p>
                <p className="text-sm text-muted-foreground">Baseline</p>
                <p className="text-xs text-foreground-subtle">{baselineWinRate.toFixed(1)}%</p>
              </div>
              <div className="bg-background-card border border-border rounded p-4 text-center">
                <p className="text-2xl text-foreground-bright">{stats.prosody}</p>
                <p className="text-sm text-muted-foreground">Prosody</p>
                <p className="text-xs text-foreground-subtle">{prosodyWinRate.toFixed(1)}%</p>
              </div>
              <div className="bg-background-card border border-border rounded p-4 text-center">
                <p className="text-2xl text-foreground">{stats.tie}</p>
                <p className="text-sm text-muted-foreground">Ties</p>
              </div>
            </div>

            {/* Conclusion */}
            <div className="bg-background-card border border-border rounded p-6">
              <h3 className="text-sm text-foreground-bright mb-2">Conclusion</h3>
              <p className="text-sm text-foreground">
                {prosodyWinRate > 55 ? (
                  <>
                    <span className="text-foreground-bright">Prosody conditioning shows improvement</span>
                    {" "}- Preferred in {prosodyWinRate.toFixed(0)}% of comparisons
                  </>
                ) : baselineWinRate > 55 ? (
                  <>
                    <span className="text-foreground">Baseline performs better</span>
                    {" "}- Preferred in {baselineWinRate.toFixed(0)}% of comparisons
                  </>
                ) : (
                  <>
                    <span className="text-muted-foreground">No significant difference</span>
                    {" "}- Results are too close to draw a conclusion
                  </>
                )}
              </p>
            </div>
          </div>
        </main>

        {/* Right Sidebar */}
        <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
          <Section title="Win Rates" defaultOpen>
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Baseline</span>
                  <span className="text-foreground">{baselineWinRate.toFixed(1)}%</span>
                </div>
                <div className="h-1 bg-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-foreground/50 transition-all"
                    style={{ width: `${baselineWinRate}%` }}
                  />
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Prosody</span>
                  <span className="text-foreground-bright">{prosodyWinRate.toFixed(1)}%</span>
                </div>
                <div className="h-1 bg-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-foreground transition-all"
                    style={{ width: `${prosodyWinRate}%` }}
                  />
                </div>
              </div>
            </div>
          </Section>
        </aside>
      </div>
    );
  }

  // No Pairs Loaded
  if (!currentPair) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center max-w-md">
          <Upload className="w-8 h-8 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-lg text-foreground-bright mb-2">No Test Pairs Loaded</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Generate audio samples first using the evaluation scripts, then
            place them in the public/audio directory.
          </p>
          <Button onClick={resetTest} variant="outline">
            Reload Test Pairs
          </Button>
        </div>
      </div>
    );
  }

  // Main Test View
  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Progress" defaultOpen>
          <div className="space-y-3">
            {/* Progress Bar */}
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-muted-foreground">Completion</span>
                <span className="text-foreground-bright">{progress.toFixed(0)}%</span>
              </div>
              <div className="h-1 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-foreground transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            <StatRow label="Pair" value={`${currentPairIndex + 1} / ${testPairs.length}`} />
            <StatRow label="Question" value={`${currentQuestionIndex + 1} / ${questions.length}`} />
            <StatRow label="Completed" value={results.length} />
          </div>
        </Section>

        <Section title="Current Test" defaultOpen>
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">Emotion</div>
            <div className="text-sm text-foreground-bright capitalize">{currentPair.emotion}</div>
          </div>
        </Section>

        <Section title="Instructions">
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">1.</span>
              Listen to both samples
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">2.</span>
              Select which is better
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">3.</span>
              Or choose "No Difference"
            </li>
          </ul>
        </Section>

        <div className="px-4 py-3 border-t border-border mt-auto">
          <button
            onClick={resetTest}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <RotateCcw className="w-3 h-3" />
            Restart Test
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
        <div className="p-8 max-w-2xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-lg text-foreground-bright mb-2">A/B Listening Test</h1>
            <p className="text-sm text-muted-foreground">
              Compare audio samples and select your preference
            </p>
          </div>

          {/* Script Text */}
          <div className="bg-background-card border border-border rounded p-6 mb-6">
            <p className="text-lg text-foreground-bright italic leading-relaxed">
              "{currentPair.text}"
            </p>
          </div>

          {/* Question */}
          <div className="text-center mb-6">
            <span className="text-sm text-foreground-bright">{currentQuestion.text}</span>
          </div>

          {/* Audio Players */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            {/* Audio A */}
            <div className="bg-background-card border border-border rounded p-6 text-center">
              <h3 className="text-sm text-foreground-bright mb-4">Sample A</h3>
              <Button
                onClick={() => (isPlaying === "A" ? stopAudio() : playAudio("A"))}
                size="lg"
                variant={isPlaying === "A" ? "default" : "outline"}
                className="w-full h-14"
              >
                {isPlaying === "A" ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Play className="w-5 h-5" />
                )}
              </Button>
              <audio
                ref={audioRefA}
                src={currentPair.audioA}
                onEnded={handleAudioEnd}
                onError={() => toast.error("Audio A not found")}
              />
            </div>

            {/* Audio B */}
            <div className="bg-background-card border border-border rounded p-6 text-center">
              <h3 className="text-sm text-foreground-bright mb-4">Sample B</h3>
              <Button
                onClick={() => (isPlaying === "B" ? stopAudio() : playAudio("B"))}
                size="lg"
                variant={isPlaying === "B" ? "default" : "outline"}
                className="w-full h-14"
              >
                {isPlaying === "B" ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Play className="w-5 h-5" />
                )}
              </Button>
              <audio
                ref={audioRefB}
                src={currentPair.audioB}
                onEnded={handleAudioEnd}
                onError={() => toast.error("Audio B not found")}
              />
            </div>
          </div>

          {/* Selection Buttons */}
          <div className="grid grid-cols-3 gap-3">
            <Button
              onClick={() => selectWinner("A")}
              variant="outline"
              className="h-12"
            >
              <Check className="w-4 h-4 mr-2" />
              Sample A
            </Button>
            <Button
              onClick={() => selectWinner("tie")}
              variant="outline"
              className="h-12"
            >
              No Difference
            </Button>
            <Button
              onClick={() => selectWinner("B")}
              variant="outline"
              className="h-12"
            >
              <Check className="w-4 h-4 mr-2" />
              Sample B
            </Button>
          </div>

          <p className="text-xs text-muted-foreground text-center mt-4">
            Listen to both samples before selecting. You can replay as needed.
          </p>
        </div>
      </main>

      {/* Right Sidebar */}
      <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Running Totals" defaultOpen>
          <div className="space-y-2">
            <StatRow
              label="Baseline"
              value={results.filter(r => r.selectedModel === "baseline").length}
            />
            <StatRow
              label="Prosody"
              value={results.filter(r => r.selectedModel === "prosody").length}
            />
            <StatRow
              label="Ties"
              value={results.filter(r => r.selectedModel === "tie").length}
            />
          </div>
        </Section>

        <Section title="Test Info">
          <div className="space-y-2">
            <StatRow label="Total Pairs" value={testPairs.length} />
            <StatRow label="Questions/Pair" value={questions.length} />
            <StatRow label="Total Tests" value={testPairs.length * questions.length} />
          </div>
        </Section>
      </aside>
    </div>
  );
}
