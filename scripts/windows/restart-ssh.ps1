# Auto-restart SSH in WSL if it's down
# This script should be run by Windows Task Scheduler every 5 minutes

$logFile = "C:\Users\doc\ssh-watchdog.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Check if SSH is running
$sshStatus = wsl bash -c "service ssh status 2>&1"

if ($sshStatus -match "not running" -or $sshStatus -match "failed") {
    Add-Content -Path $logFile -Value "[$timestamp] SSH down - restarting..."
    wsl bash -c "sudo service ssh start"
    Start-Sleep -Seconds 2

    # Verify it started
    $newStatus = wsl bash -c "service ssh status 2>&1"
    if ($newStatus -match "running") {
        Add-Content -Path $logFile -Value "[$timestamp] SSH restarted successfully"
    } else {
        Add-Content -Path $logFile -Value "[$timestamp] SSH restart FAILED: $newStatus"
    }
} else {
    # Only log every hour to avoid spam
    $lastLog = Get-Content $logFile -Tail 1 -ErrorAction SilentlyContinue
    if (-not $lastLog -or $lastLog -notmatch (Get-Date -Format "yyyy-MM-dd HH:")) {
        Add-Content -Path $logFile -Value "[$timestamp] SSH running OK"
    }
}
