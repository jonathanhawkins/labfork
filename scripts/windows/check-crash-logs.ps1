# Check Windows crash/reboot logs
# Run this via Parsec to diagnose if machine is crashing

Write-Host "=== Windows System Crash & Reboot Analysis ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check unexpected shutdowns/reboots
Write-Host "1. Recent Unexpected Shutdowns (Event ID 6008):" -ForegroundColor Yellow
Get-EventLog -LogName System -Source "EventLog" -After (Get-Date).AddDays(-2) |
    Where-Object { $_.EventID -eq 6008 } |
    Select-Object TimeGenerated, Message |
    Format-Table -AutoSize

# 2. Check system crashes (bugchecks/BSODs)
Write-Host "2. System Crashes / BSODs (Event ID 1001):" -ForegroundColor Yellow
Get-EventLog -LogName System -Source "BugCheck" -After (Get-Date).AddDays(-2) -ErrorAction SilentlyContinue |
    Select-Object TimeGenerated, Message |
    Format-Table -AutoSize

# 3. Check kernel power events (unexpected power loss)
Write-Host "3. Unexpected Power Events (Event ID 41):" -ForegroundColor Yellow
Get-EventLog -LogName System -Source "Microsoft-Windows-Kernel-Power" -After (Get-Date).AddDays(-2) |
    Where-Object { $_.EventID -eq 41 } |
    Select-Object TimeGenerated, EntryType, Message |
    Format-Table -AutoSize

# 4. Check clean shutdowns vs dirty shutdowns
Write-Host "4. All Shutdown/Startup Events (Last 20):" -ForegroundColor Yellow
Get-EventLog -LogName System -After (Get-Date).AddDays(-2) |
    Where-Object { $_.EventID -in @(1074, 1076, 6005, 6006, 6008, 6009, 41) } |
    Select-Object TimeGenerated, EventID, Source, Message |
    Sort-Object TimeGenerated -Descending |
    Select-Object -First 20 |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Event ID Reference:" -ForegroundColor Cyan
Write-Host "  1074 - Clean shutdown initiated by user/system"
Write-Host "  6005 - Event Log service started (boot)"
Write-Host "  6006 - Event Log service stopped (shutdown)"
Write-Host "  6008 - Unexpected shutdown (dirty)"
Write-Host "  6009 - OS version on boot"
Write-Host "    41 - System rebooted without cleanly shutting down (crash/power loss)"
Write-Host ""

# 5. Current uptime
Write-Host "5. Current System Uptime:" -ForegroundColor Yellow
$os = Get-WmiObject -Class Win32_OperatingSystem
$uptime = (Get-Date) - $os.ConvertToDateTime($os.LastBootUpTime)
Write-Host "  Last Boot: $($os.ConvertToDateTime($os.LastBootUpTime))"
Write-Host "  Uptime: $($uptime.Days) days, $($uptime.Hours) hours, $($uptime.Minutes) minutes"
Write-Host ""

# 6. Check WSL status
Write-Host "6. WSL Status:" -ForegroundColor Yellow
$wslProcesses = Get-Process -Name "wsl*", "bash", "docker" -ErrorAction SilentlyContinue
if ($wslProcesses) {
    Write-Host "  WSL processes running: $($wslProcesses.Count)"
} else {
    Write-Host "  No WSL processes found"
}

# 7. Check if SSH service is running in WSL
Write-Host ""
Write-Host "7. SSH Service in WSL:" -ForegroundColor Yellow
wsl bash -c "service ssh status 2>&1"

Write-Host ""
Write-Host "=== Analysis Complete ===" -ForegroundColor Cyan
