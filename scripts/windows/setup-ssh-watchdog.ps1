# Setup SSH Watchdog - Run this ONCE via Parsec/RDP as Administrator
# Creates a Windows Task Scheduler job to restart SSH every 5 minutes if down

Write-Host "Setting up SSH Watchdog for WSL..." -ForegroundColor Cyan

# 1. Copy restart script to Windows user directory
$scriptPath = "C:\Users\doc\restart-ssh.ps1"
$sourceScript = "\\wsl.localhost\Ubuntu\home\doc\dev\voice-clone-pipeline\scripts\windows\restart-ssh.ps1"

if (Test-Path $sourceScript) {
    Copy-Item -Path $sourceScript -Destination $scriptPath -Force
    Write-Host "✓ Copied restart-ssh.ps1 to $scriptPath" -ForegroundColor Green
} else {
    Write-Host "✗ Source script not found at $sourceScript" -ForegroundColor Red
    Write-Host "  Manually copy scripts/windows/restart-ssh.ps1 from WSL to C:\Users\doc\" -ForegroundColor Yellow
    exit 1
}

# 2. Create Task Scheduler job
Write-Host "`nCreating Task Scheduler job..." -ForegroundColor Cyan

$taskName = "WSL-SSH-Watchdog"

# Remove existing task if present
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "  Removed existing task" -ForegroundColor Yellow
}

# Create new task
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Description "Auto-restart SSH in WSL if down"

Write-Host "✓ Task '$taskName' created" -ForegroundColor Green

# 3. Test run
Write-Host "`nRunning initial test..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3

# 4. Show status
Write-Host "`nVerifying SSH is running..." -ForegroundColor Cyan
$sshStatus = wsl bash -c "service ssh status 2>&1"
if ($sshStatus -match "running") {
    Write-Host "✓ SSH is running" -ForegroundColor Green
} else {
    Write-Host "✗ SSH status: $sshStatus" -ForegroundColor Red
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Task: $taskName" -ForegroundColor White
Write-Host "Runs: Every 5 minutes" -ForegroundColor White
Write-Host "Log: C:\Users\doc\ssh-watchdog.log" -ForegroundColor White
Write-Host "`nTo view log:" -ForegroundColor Yellow
Write-Host "  Get-Content C:\Users\doc\ssh-watchdog.log -Tail 20" -ForegroundColor Gray
