import { NextResponse } from "next/server";
import {
  readFileSync,
  readdirSync,
  existsSync,
  statSync,
  writeFileSync,
  mkdirSync,
} from "fs";
import { join } from "path";
import { homedir } from "os";

// Read Claude Code task list from ~/.claude/tasks/{session-id}/
// Tasks are stored as individual JSON files per task

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
const AGENT_STATE_URL = process.env.AGENT_STATE_URL || '';

interface Task {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  owner?: string;
  blockedBy?: string[];
  blocks?: string[];
  activeForm?: string;
  createdAt?: string;
  updatedAt?: string;
}

interface SessionTasks {
  sessionId: string;
  tasks: Task[];
  modifiedAt: Date;
}

export async function GET() {
  try {
    if (BACKEND_URL) {
      try {
        const [tasksRes, agentsRes] = await Promise.all([
          fetch(`${BACKEND_URL}/api/lab/tasks`, { cache: "no-store" }),
          fetch(`${BACKEND_URL}/api/lab/research-agents`, { cache: "no-store" }),
        ]);

        if (tasksRes.ok) {
          const tasksData = await tasksRes.json();
          const agentsData = agentsRes.ok ? await agentsRes.json() : { agents: {} };
          return NextResponse.json({
            tasks: tasksData.tasks || [],
            agents: agentsData.agents || {},
            sessionId: tasksData.sessionId || "voice-clone-pipeline",
            taskListId: tasksData.taskListId || "voice-clone-pipeline",
          });
        }
      } catch (error) {
        console.error("Backend task fetch failed, falling back to local tasks:", error);
      }
    }

    // Check if we're on Vercel and should use remote data
    const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;

    if (isVercel && AGENT_STATE_URL) {
      try {
        const [tasksRes, agentsRes] = await Promise.all([
          fetch(`${AGENT_STATE_URL}/tasks`, {
            cache: 'no-store',
            signal: AbortSignal.timeout(5000),
          }),
          fetch(`${AGENT_STATE_URL}/agents`, {
            cache: 'no-store',
            signal: AbortSignal.timeout(5000),
          }),
        ]);

        if (tasksRes.ok) {
          const tasksData = await tasksRes.json();
          const agentsData = agentsRes.ok ? await agentsRes.json() : {};
          return NextResponse.json({
            tasks: tasksData.tasks || [],
            agents: agentsData,
            sessionId: "voice-clone-pipeline",
            taskListId: "voice-clone-pipeline",
          });
        }
      } catch (error) {
        console.error("[Tasks] Remote fetch failed:", error);
      }
      // Return empty if remote fetch failed
      return NextResponse.json({
        tasks: [],
        agents: {},
        sessionId: "voice-clone-pipeline",
        taskListId: "voice-clone-pipeline",
      });
    }

    const tasksDir = join(homedir(), ".claude", "tasks");
    let allTasks: Task[] = [];
    let latestSession = "";

    // Find all session directories and get tasks from the most recent ones
    if (existsSync(tasksDir)) {
      const sessions = readdirSync(tasksDir).filter((name) => {
        const fullPath = join(tasksDir, name);
        return statSync(fullPath).isDirectory() && !name.startsWith(".");
      });

      // Get tasks from each session with modification time
      const sessionTasks: SessionTasks[] = sessions
        .map((sessionId) => {
          const sessionDir = join(tasksDir, sessionId);
          try {
            const taskFiles = readdirSync(sessionDir).filter(
              (f) => f.endsWith(".json") && !f.startsWith(".")
            );

            const tasks = taskFiles.map((file) => {
              const content = readFileSync(join(sessionDir, file), "utf-8");
              return JSON.parse(content) as Task;
            });

            const stat = statSync(sessionDir);
            return {
              sessionId,
              tasks,
              modifiedAt: stat.mtime,
            };
          } catch {
            return { sessionId, tasks: [], modifiedAt: new Date(0) };
          }
        })
        .filter((s) => s.tasks.length > 0)
        .sort((a, b) => b.modifiedAt.getTime() - a.modifiedAt.getTime());

      // Get tasks from the most recently modified session
      if (sessionTasks.length > 0) {
        allTasks = sessionTasks[0].tasks;
        latestSession = sessionTasks[0].sessionId;
      }
    }

    // Also get Research Manager agent state
    // Note: process.cwd() is frontend/, so go up one level to project root
    const projectRoot = join(process.cwd(), "..");
    const agentStatePath = join(
      projectRoot,
      ".skills",
      "research-manager",
      "state",
      "agents.json"
    );

    let agents: Record<string, any> = {};
    if (existsSync(agentStatePath)) {
      try {
        agents = JSON.parse(readFileSync(agentStatePath, "utf-8"));
      } catch (e) {
        console.error("Error reading agents.json:", e);
      }
    }

    return NextResponse.json({
      tasks: allTasks,
      agents,
      sessionId: latestSession,
      taskListId: "voice-clone-pipeline",
    });
  } catch (error) {
    console.error("Error fetching tasks:", error);
    return NextResponse.json(
      { error: "Failed to fetch tasks", tasks: [], agents: {} },
      { status: 500 }
    );
  }
}

// Create a new task
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { subject, description } = body;

    if (!subject) {
      return NextResponse.json(
        { error: "Subject is required" },
        { status: 400 }
      );
    }

    if (BACKEND_URL) {
      try {
        const response = await fetch(`${BACKEND_URL}/api/lab/tasks`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ subject, description }),
        });
        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }
      } catch (error) {
        console.error("Backend task create failed, falling back to local tasks:", error);
      }
    }

    const tasksDir = join(homedir(), ".claude", "tasks");

    // Find the most recent session directory or create one
    let targetSessionDir = "";
    let nextTaskId = 1;

    if (existsSync(tasksDir)) {
      const sessions = readdirSync(tasksDir)
        .filter((name) => {
          const fullPath = join(tasksDir, name);
          return (
            statSync(fullPath).isDirectory() &&
            !name.startsWith(".") &&
            name !== "TASKS-ALIGNED"
          );
        })
        .map((name) => ({
          name,
          mtime: statSync(join(tasksDir, name)).mtime,
        }))
        .sort((a, b) => b.mtime.getTime() - a.mtime.getTime());

      if (sessions.length > 0) {
        targetSessionDir = join(tasksDir, sessions[0].name);

        // Find the next task ID
        const existingTasks = readdirSync(targetSessionDir).filter(
          (f) => f.endsWith(".json") && !f.startsWith(".")
        );
        const taskIds = existingTasks.map((f) =>
          parseInt(f.replace(".json", ""), 10)
        );
        nextTaskId = Math.max(0, ...taskIds) + 1;
      }
    }

    // If no session exists, create a new one
    if (!targetSessionDir) {
      const newSessionId = `user-tasks-${Date.now()}`;
      targetSessionDir = join(tasksDir, newSessionId);
      mkdirSync(targetSessionDir, { recursive: true });
    }

    // Create the new task
    const newTask: Task = {
      id: String(nextTaskId),
      subject,
      description: description || "",
      status: "pending",
      activeForm: "",
      blocks: [],
      blockedBy: [],
    };

    const taskPath = join(targetSessionDir, `${nextTaskId}.json`);
    writeFileSync(taskPath, JSON.stringify(newTask, null, 2));

    return NextResponse.json({
      success: true,
      task: newTask,
    });
  } catch (error) {
    console.error("Error creating task:", error);
    return NextResponse.json(
      { error: "Failed to create task" },
      { status: 500 }
    );
  }
}

// Update a task
export async function PATCH(request: Request) {
  try {
    const body = await request.json();
    const { id, status, subject, description } = body;

    if (!id) {
      return NextResponse.json({ error: "Task ID is required" }, { status: 400 });
    }

    if (BACKEND_URL) {
      try {
        const response = await fetch(`${BACKEND_URL}/api/lab/tasks`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, status, subject, description }),
        });
        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }
      } catch (error) {
        console.error("Backend task update failed, falling back to local tasks:", error);
      }
    }

    const tasksDir = join(homedir(), ".claude", "tasks");

    // Find the task in recent sessions
    if (existsSync(tasksDir)) {
      const sessions = readdirSync(tasksDir)
        .filter((name) => {
          const fullPath = join(tasksDir, name);
          return statSync(fullPath).isDirectory() && !name.startsWith(".");
        })
        .sort((a, b) => {
          const aTime = statSync(join(tasksDir, a)).mtime;
          const bTime = statSync(join(tasksDir, b)).mtime;
          return bTime.getTime() - aTime.getTime();
        });

      for (const sessionId of sessions) {
        const taskPath = join(tasksDir, sessionId, `${id}.json`);
        if (existsSync(taskPath)) {
          const task = JSON.parse(readFileSync(taskPath, "utf-8")) as Task;

          // Update fields
          if (status) task.status = status;
          if (subject) task.subject = subject;
          if (description !== undefined) task.description = description;

          writeFileSync(taskPath, JSON.stringify(task, null, 2));

          return NextResponse.json({ success: true, task });
        }
      }
    }

    return NextResponse.json({ error: "Task not found" }, { status: 404 });
  } catch (error) {
    console.error("Error updating task:", error);
    return NextResponse.json(
      { error: "Failed to update task" },
      { status: 500 }
    );
  }
}
