# WSL2 Ollama Networking Troubleshooting

## Problem: Research Agents Get Stuck / Can't Connect to Ollama

### Symptoms

- Research orchestrator shows "No viable tasks (all blocked or max retries)"
- Agent tmux sessions spawn but immediately fail with "stuck" status after ~10 seconds
- Tasks hit max retry count (4 retries) quickly
- Agent logs show Ollama connection errors like:
  ```
  curl: (7) Failed to connect to 192.168.128.1:11434
  curl: (52) Empty reply from server
  Connection reset by peer
  ```
- Firefly research tasks (#371-375) stuck in "in_progress" status with old owners

### Root Cause

**Ollama runs on Windows and listens only on `127.0.0.1:11434` (Windows localhost)**, which WSL2 cannot access because:

1. WSL2 uses a separate network namespace (different from WSL1)
2. Windows `127.0.0.1` is not the same as WSL2's `127.0.0.1`
3. WSL2 can access Windows services via the Windows host IP (like `192.168.128.1`), BUT:
   - Ollama by default only binds to `127.0.0.1`, not `0.0.0.0`
   - Even with port forwarding, if the service isn't listening on the right interface, it won't work

## Solution: Enable localhost Forwarding

The cleanest solution is to enable `localhostForwarding` in WSL2, which automatically forwards `localhost:*` requests from WSL2 to Windows localhost.

### Step 1: Configure .wslconfig

**On Windows**, create or edit `C:\Users\Doc Holiday\.wslconfig`:

```ini
[wsl2]
localhostForwarding=true
```

**IMPORTANT**: Do NOT add `networkingMode=mirrored` on Windows 10. Mirrored networking:
- Only works on Windows 11 22H2+
- Has known SSH bugs even on Windows 11
- Will break SSH connectivity

### Step 2: Restart WSL2

**On Windows**, open PowerShell and run:

```powershell
wsl --shutdown
```

Wait 10 seconds, then start WSL again by opening your Ubuntu terminal or running:

```powershell
wsl
```

### Step 3: Verify Ollama is Accessible from WSL2

**In WSL2**, test the connection:

```bash
# This should now work via localhost
curl http://localhost:11434/api/version

# Expected output:
{"version":"0.15.2"}
```

If this works, Ollama is now accessible from WSL2!

## Alternative Solution: Port Forwarding (If localhost forwarding doesn't work)

If `localhostForwarding=true` doesn't work for some reason, you can use Windows port proxy:

### Option A: Port Proxy to Windows Host IP

**On Windows**, run in PowerShell as Administrator:

```powershell
# Get Windows host IP (the IP WSL2 sees Windows as)
wsl hostname -I  # Note: this shows WSL IP, we need Windows IP from WSL perspective

# Add port proxy from all interfaces to localhost
netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=127.0.0.1

# Allow through firewall
netsh advfirewall firewall add rule name="Ollama WSL Access" dir=in action=allow protocol=TCP localport=11434

# Verify
netsh interface portproxy show all
```

Then from WSL2, get the Windows host IP:

```bash
# Get Windows IP from WSL2's perspective
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo $WIN_IP

# Test Ollama via Windows host IP
curl http://$WIN_IP:11434/api/version
```

### Option B: Configure Ollama to Listen on 0.0.0.0

**On Windows**, set environment variable before starting Ollama:

```powershell
# Set for current session
$env:OLLAMA_HOST = "0.0.0.0:11434"

# Set permanently (Machine level)
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'Machine')

# Restart Ollama
taskkill /F /IM ollama.exe
Start-Process -FilePath "C:\Users\Doc Holiday\AppData\Local\Programs\Ollama\ollama.exe" -ArgumentList "serve"
```

Verify it's listening on all interfaces:

```powershell
netstat -ano | findstr 11434
# Should show: TCP 0.0.0.0:11434 (not 127.0.0.1:11434)
```

## Reset Stuck Research Tasks

After fixing Ollama connectivity, you need to reset tasks that failed before the fix.

### Step 1: Reset Retry Counts

**On the 4090 (via SSH)**:

```bash
ssh doc@$REMOTE_GPU_HOST "cd ~/dev/labfork/.skills/research-manager && node reset-retries.js 371 && node reset-retries.js 372 && node reset-retries.js 373 && node reset-retries.js 374 && node reset-retries.js 375"
```

### Step 2: Reset Task Status to Pending

Tasks stuck in "in_progress" with old owners won't be picked up by the orchestrator. Reset them:

```bash
ssh doc@$REMOTE_GPU_HOST 'cat > /tmp/reset-tasks.js << '\''EOF'\''
const fs = require("fs");
const path = require("path");

const tasksDir = path.join(process.env.HOME, ".claude", "tasks", "labfork");
const taskIds = process.argv.slice(2);

for (const taskId of taskIds) {
  const file = path.join(tasksDir, `${taskId}.json`);
  try {
    const task = JSON.parse(fs.readFileSync(file, "utf-8"));
    task.status = "pending";
    task.owner = null;
    fs.writeFileSync(file, JSON.stringify(task, null, 2));
    console.log(`Reset task #${taskId} to pending`);
  } catch (e) {
    console.error(`Failed to reset task #${taskId}:`, e.message);
  }
}
EOF
node /tmp/reset-tasks.js 371 372 373 374 375'
```

### Step 3: Restart Orchestrator

```bash
ssh doc@$REMOTE_GPU_HOST "cd ~/dev/labfork && pkill -f orchestrator.js && sleep 3 && nohup node .skills/research-manager/orchestrator.js > .skills/research-manager/state/orchestrator.log 2>&1 &"
```

### Step 4: Verify Agents Are Running

Wait 30 seconds for orchestrator to spawn agents, then check:

```bash
# Check orchestrator log
ssh doc@$REMOTE_GPU_HOST "tail -50 ~/dev/labfork/.skills/research-manager/state/orchestrator.log"

# Check for agent tmux sessions
ssh doc@$REMOTE_GPU_HOST "tmux ls | grep rm-task"

# Should see agents like:
# rm-task-371-1769735614325: 1 windows (created ...)
# rm-task-372-1769735701527: 1 windows (created ...)
```

## Verification Checklist

After applying the fix:

- [ ] `curl http://localhost:11434/api/version` works from WSL2
- [ ] Orchestrator log shows "Spawned agent" messages
- [ ] Agent tmux sessions are created: `tmux ls | grep rm-task`
- [ ] Agents stay running for more than 10 seconds (not immediately stuck)
- [ ] Research output files appear in `docs/firefly/` within 30 minutes

## Prevention

To avoid this issue in the future:

1. **Keep `localhostForwarding=true` in `.wslconfig`** - this is the most reliable solution for Windows 10
2. **Don't use `networkingMode=mirrored` on Windows 10** - it doesn't work and breaks SSH
3. **After WSL restarts**, verify Ollama is still accessible: `curl http://localhost:11434/api/version`
4. **Monitor orchestrator logs** for connection errors: `tail -f .skills/research-manager/state/orchestrator.log`

## Common Errors and Fixes

### Error: "Connection refused" from WSL2

**Cause**: Ollama not running on Windows, or Windows firewall blocking

**Fix**:
```powershell
# On Windows - check if Ollama is running
tasklist | findstr ollama

# If not running, start it
ollama serve
```

### Error: "Empty reply from server"

**Cause**: Ollama listening on wrong interface (IPv6 only, or 127.0.0.1 only)

**Fix**: Enable `localhostForwarding=true` in `.wslconfig` (Step 1 above)

### Error: Tasks show "4 retries" and won't run

**Cause**: Tasks hit max retry limit before Ollama was fixed

**Fix**: Use `reset-retries.js` script (see "Reset Stuck Research Tasks" above)

### Error: Tasks stuck in "in_progress" status

**Cause**: Old agents claimed the tasks but got stuck/killed, owners not cleared

**Fix**: Use `reset-tasks.js` script to set status=pending and owner=null (see Step 2 above)

## Related Issues

- **SSH connectivity**: See CLAUDE.md section "Troubleshooting: SSH Connection Fails After Reboot"
- **Port forwarding for other services**: If you need to expose other WSL2 services (like the frontend on port 3003), use the same port proxy approach shown above

## Technical Details

### Why localhost forwarding works

With `localhostForwarding=true`:
- WSL2 kernel automatically forwards `localhost:*` requests to Windows
- No need to know Windows host IP (which can change)
- No need for manual port proxy configuration
- Works for all ports automatically

### WSL2 Networking Architecture

```
┌─────────────────────────────────────┐
│         Windows (Host)              │
│  - Ollama on 127.0.0.1:11434       │
│  - Tailscale on $REMOTE_GPU_HOST      │
└─────────────────────────────────────┘
              │
              │ localhostForwarding=true
              │ (forwards localhost:* → Windows localhost:*)
              ▼
┌─────────────────────────────────────┐
│         WSL2 (Namespace)            │
│  - SSH on WSL_IP:22                │
│  - Research agents                  │
│  - curl localhost:11434 → Windows  │
└─────────────────────────────────────┘
```

Without `localhostForwarding=true`:
- WSL2's `localhost` ≠ Windows's `localhost`
- Must use Windows host IP (e.g., `192.168.128.1`)
- Requires Ollama to bind to `0.0.0.0` or port forwarding
