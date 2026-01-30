# Setup WSL2 + SSH Auto-Start on Windows Boot
# Run this ONCE as Administrator via Parsec

Write-Host "Setting up WSL2 SSH Auto-Start..." -ForegroundColor Cyan

# 1. Enable systemd in WSL2 (if not already enabled)
Write-Host "`n1. Enabling systemd in WSL2..." -ForegroundColor Yellow
wsl bash -c "sudo mkdir -p /etc && sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
systemd=true
EOF
echo 'Systemd enabled in /etc/wsl.conf'"

# 2. Restart WSL to apply systemd
Write-Host "`n2. Restarting WSL to apply changes..." -ForegroundColor Yellow
wsl --shutdown
Start-Sleep -Seconds 5

# 3. Start WSL and enable SSH service
Write-Host "`n3. Enabling SSH service in WSL..." -ForegroundColor Yellow
wsl bash -c "sudo systemctl enable ssh && sudo systemctl start ssh && echo 'SSH service enabled and started'"

# 4. Create startup script on Windows side
Write-Host "`n4. Creating Windows startup script..." -ForegroundColor Yellow
$startupScript = @"
@echo off
REM Start WSL2 and ensure SSH is running
wsl bash -c "sudo service ssh start 2>/dev/null || sudo systemctl start ssh 2>/dev/null"
"@

$startupScriptPath = "C:\Users\doc\start-wsl-ssh.bat"
Set-Content -Path $startupScriptPath -Value $startupScript
Write-Host "  Created: $startupScriptPath" -ForegroundColor Green

# 5. Create Task Scheduler job to run at startup
Write-Host "`n5. Creating Task Scheduler job..." -ForegroundColor Yellow
$taskName = "WSL-SSH-AutoStart"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create new task
$action = New-ScheduledTaskAction -Execute "C:\Windows\System32\cmd.exe" -Argument "/c `"$startupScriptPath`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Auto-start WSL2 and SSH service on Windows boot"

Write-Host "  Task '$taskName' created successfully" -ForegroundColor Green

# 6. Test it now
Write-Host "`n6. Testing SSH service..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
$sshStatus = wsl bash -c "service ssh status 2>&1"
if ($sshStatus -match "running") {
    Write-Host "  ✓ SSH is running!" -ForegroundColor Green
} else {
    Write-Host "  ✗ SSH status: $sshStatus" -ForegroundColor Red
    Write-Host "  Trying to start manually..." -ForegroundColor Yellow
    wsl bash -c "sudo service ssh start"
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "SSH will now auto-start when Windows boots." -ForegroundColor White
Write-Host ""
Write-Host "To verify after reboot:" -ForegroundColor Yellow
Write-Host "  wsl bash -c 'service ssh status'" -ForegroundColor Gray
