# PowerShell script to start the FastAPI server
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists and activate it
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Start the server with reload enabled for development
Write-Host "Starting Uvicorn server on http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

