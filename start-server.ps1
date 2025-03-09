# PowerShell script to start the Flask server as a background job

# Start the Flask server as a background job
$job = Start-Job -ScriptBlock {
    Set-Location -Path $using:PWD
    python run_local.py
}

Write-Host "Flask server started as background job (ID: $($job.Id))"
Write-Host "Server is running at http://127.0.0.1:5000"
Write-Host ""
Write-Host "To view server output: Receive-Job -Id $($job.Id)"
Write-Host "To stop the server: Stop-Job -Id $($job.Id); Remove-Job -Id $($job.Id)"
Write-Host ""
Write-Host "You can now use this terminal for Git operations." 