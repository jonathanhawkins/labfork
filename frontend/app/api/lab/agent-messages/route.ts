import { NextResponse } from "next/server";
import { readFileSync, readdirSync, existsSync, statSync } from "fs";
import { join } from "path";

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;

// Paths
const projectRoot = join(process.cwd(), "..");
const OUTPUTS_DIR = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "outputs"
);

interface AgentMessage {
  agent: string;
  message: string;
  timestamp: Date;
  type: "output" | "action" | "thinking" | "complete";
}

/**
 * Strip ANSI escape codes from terminal output
 */
function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str
    // CSI sequences (most common ANSI codes)
    .replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, "")
    // OSC sequences (operating system commands)
    .replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, "")
    // Other escape sequences
    .replace(/\x1b[PX^_][^\x1b]*\x1b\\/g, "")
    .replace(/\x1b[@-Z\\-_]/g, "")
    // Unicode escape chars that might slip through
    .replace(/\u001b\[[0-9;?]*[a-zA-Z]/g, "")
    .replace(/\u001b\][^\u0007]*\u0007/g, "")
    // Control characters (keep newlines)
    .replace(/[\x00-\x09\x0b-\x1a\x1c-\x1f]/g, "")
    // Clean up any remaining junk
    .replace(/\[[\?0-9;]*[a-zA-Z]/g, "");
}

/**
 * Parse agent output to extract meaningful messages
 */
function parseAgentOutput(content: string, agentName: string): AgentMessage[] {
  const messages: AgentMessage[] = [];
  // Strip ANSI codes before parsing
  const cleanContent = stripAnsi(content);
  const lines = cleanContent.split("\n");

  // Track last timestamp
  let lastTimestamp = new Date();

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip empty lines and noise
    if (!trimmed) continue;
    if (trimmed.length < 15) continue;

    // Skip lines with leftover escape sequences
    if (trimmed.includes("\\u001b") || trimmed.includes("\u001b")) continue;
    if (/^\[[\?0-9;]*[a-zA-Z]/.test(trimmed)) continue;

    // Skip decorative/structural lines (including table borders)
    if (/^[─═▐▛▜▘▝\-\s⎿┌┐└┘├┤┬┴┼│]+$/.test(trimmed)) continue;
    // Skip table row lines with just dividers
    if (/^[│├┤┬┴┼─]+$/.test(trimmed)) continue;
    if (trimmed.startsWith("Agent:")) continue;
    if (trimmed.startsWith("Type:")) continue;
    if (trimmed.startsWith("Working Dir:")) continue;
    if (trimmed.startsWith("Session:")) continue;
    if (trimmed.includes("Claude Code v")) continue;
    if (trimmed.includes("~/dev/")) continue;
    if (trimmed.includes("Opus 4.5")) continue;
    if (trimmed.includes("ctrl+")) continue;
    if (trimmed.includes("more tool uses")) continue;
    if (trimmed.includes("bypass permissions")) continue;
    if (trimmed.includes("Checking for updates")) continue;
    if (trimmed.includes("shift+Tab")) continue;
    if (trimmed.startsWith("❯")) continue;
    if (/^[0-9]+[A-Z]/.test(trimmed)) continue; // Leftover ANSI like "81H"
    if (/^\d+✔/.test(trimmed)) continue; // Task list artifacts like "81✔"
    if (trimmed.includes("activitylog")) continue; // Merged words from rendering
    if (/\+\d+\s+pending,\s+\d+\s+completed/.test(trimmed)) continue; // Task count lines
    if (/^\(claude\)/.test(trimmed)) continue; // Stray owner tags

    // Extract meaningful patterns
    let messageType: AgentMessage["type"] = "output";
    let messageText = trimmed;

    // Clean common spinner/status characters at start
    let cleanedLine = trimmed.replace(/^[✶✻✲✳✴✵✷✸✹✺✼✽·•◦◼◻]+\s*/, "");
    // Remove leading task list numbers like "26◼"
    cleanedLine = cleanedLine.replace(/^\d+[◼◻✔✶]\s*/, "");
    // Remove trailing timing/status info
    cleanedLine = cleanedLine.replace(/\s*\((?:Esc to interrupt)?\s*·?[^)]*(?:\d+[ms]|tok)[^)]*\)$/i, "");
    cleanedLine = cleanedLine.replace(/\s+\d+$/, ""); // Remove trailing numbers
    // Remove truncation indicators
    cleanedLine = cleanedLine.replace(/^\s*…\s*/, "");
    cleanedLine = cleanedLine.replace(/\s*…\s*$/, "...");
    // Clean table box drawing characters - extract cell content
    if (cleanedLine.includes("│")) {
      const cells = cleanedLine.split("│").map(c => c.trim()).filter(c => c.length > 0);
      if (cells.length >= 2) {
        cleanedLine = cells.join(" | ");
      }
    }

    // Claude response marker - main output
    if (cleanedLine.startsWith("⏺")) {
      messageType = "output";
      messageText = cleanedLine.replace(/^⏺\s*/, "");
    }
    // Active task line (often starts with activity description after spinner removal)
    else if (cleanedLine.includes("Implementing") || cleanedLine.includes("Researching") ||
             cleanedLine.includes("Analyzing") || cleanedLine.includes("Processing")) {
      messageType = "action";
      messageText = cleanedLine;
    }
    // Task status updates
    else if (trimmed.includes("Task #") && trimmed.includes("updated:")) {
      messageType = "complete";
      const match = trimmed.match(/Task #(\d+).*updated:\s*(.+)/);
      if (match) {
        messageText = `Task #${match[1]} updated: ${match[2]}`;
      }
    }
    // Tool uses
    else if (trimmed.match(/^(Read|Edit|Write|Bash|Grep|Glob|Task)\(/)) {
      messageType = "action";
      const toolMatch = trimmed.match(/^(\w+)\(([^)]*)\)/);
      if (toolMatch) {
        messageText = `Using ${toolMatch[1]}: ${toolMatch[2].substring(0, 80)}`;
      }
    }
    // Explore agent
    else if (trimmed.includes("Explore(")) {
      messageType = "action";
      const match = trimmed.match(/Explore\(([^)]+)\)/);
      messageText = match ? `Exploring: ${match[1]}` : "Exploring codebase";
    }
    // Task checkmarks and status
    else if (trimmed.startsWith("✔") || trimmed.startsWith("◼") || trimmed.startsWith("◻")) {
      continue; // Skip task list display lines
    }
    // Skip lines that are just task list headers
    else if (trimmed.match(/^\d+ tasks \(\d+ done/)) {
      continue;
    }
    // Skip prompt/instruction lines
    else if (trimmed.includes("INSTRUCTIONS:") || trimmed.includes("TASK #")) {
      continue;
    }
    // Regular text output - use cleanedLine for meaningful content
    else if (cleanedLine.length > 20 && !cleanedLine.includes("⎿")) {
      messageType = "output";
      messageText = cleanedLine;
    } else {
      continue; // Skip other short/structural lines
    }

    // Final validation - skip if too short or has weird chars
    if (messageText.length < 10 || messageText.length > 500) continue;
    if (/^[\s◼◻✔✶⏺⎿─═]+$/.test(messageText)) continue;
    // Skip lines that are just separators or formatting
    if (/^[─═\-\s]+$/.test(messageText)) continue;

    messages.push({
      agent: agentName,
      message: messageText.substring(0, 200),
      timestamp: lastTimestamp,
      type: messageType,
    });
  }

  // Deduplicate by message content (keep most recent)
  const seen = new Set<string>();
  const deduped: AgentMessage[] = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const key = messages[i].message.toLowerCase().substring(0, 50);
    if (!seen.has(key)) {
      seen.add(key);
      deduped.unshift(messages[i]);
    }
  }

  // Return only recent messages (last 15)
  return deduped.slice(-15);
}

/**
 * GET /api/lab/agent-messages
 * Returns recent messages from all running agents
 */
export async function GET(request: Request) {
  try {
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/agent-messages${url.search}`, {
          cache: "no-store",
        });
        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }
        const text = await response.text();
        return NextResponse.json(
          { error: text || "Backend error" },
          { status: response.status }
        );
      } catch (error) {
        console.error("[Agent Messages] Backend fetch failed, falling back to local:", error);
      }
    }

    if (!existsSync(OUTPUTS_DIR)) {
      return NextResponse.json({ messages: [] });
    }

    const logFiles = readdirSync(OUTPUTS_DIR).filter((f) => f.endsWith(".log"));
    const allMessages: AgentMessage[] = [];

    for (const logFile of logFiles) {
      const filePath = join(OUTPUTS_DIR, logFile);
      const stats = statSync(filePath);

      // Only read files modified in last hour
      const oneHourAgo = Date.now() - 60 * 60 * 1000;
      if (stats.mtime.getTime() < oneHourAgo) continue;

      try {
        // Read last 50000 chars - terminal logs have lots of ANSI overhead
        const content = readFileSync(filePath, "utf-8");
        const recentContent = content.slice(-50000);

        const agentName = logFile.replace(".log", "");
        const messages = parseAgentOutput(recentContent, agentName);
        allMessages.push(...messages);
      } catch (e) {
        // Skip unreadable files
      }
    }

    // Sort by timestamp and return most recent
    allMessages.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    return NextResponse.json({
      messages: allMessages.slice(0, 20),
      count: allMessages.length,
    });
  } catch (error) {
    console.error("[Agent Messages] Error:", error);
    return NextResponse.json({ error: "Failed to fetch messages" }, { status: 500 });
  }
}
