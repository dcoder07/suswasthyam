# PowerShell script to push the fix to GitHub

# Configure Git to not use a pager
$env:GIT_PAGER = ""

Write-Host "Starting deployment fix push..." -ForegroundColor Green

# Add the files
Write-Host "Adding files..." -ForegroundColor Cyan
git add api/index.py api/__init__.py vercel.json requirements.txt .vercelignore

# Commit the changes
Write-Host "Committing changes..." -ForegroundColor Cyan
git commit -m "Fix 404 error with standard Vercel serverless pattern"

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push suswasthyam master

Write-Host "Done! Changes pushed to GitHub. Vercel should rebuild automatically." -ForegroundColor Green
Write-Host "After deployment completes, check: https://suswasthyam.vercel.app/" -ForegroundColor Yellow

# Keep the window open
Write-Host "Press Enter to close..." -ForegroundColor Gray
Read-Host 