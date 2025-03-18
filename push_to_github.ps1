# PowerShell script to push changes to GitHub

# Configure Git to not use a pager
$env:GIT_PAGER = "cat"

# Push to the suswasthyam repository
git push suswasthyam master

# Output success message
Write-Host "Push completed successfully!" 