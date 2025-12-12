@echo off
echo Starting FastAPI server...
echo.

REM Check if virtual environment exists and activate it
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Start the server with reload enabled for development
echo Starting Uvicorn server on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

pause

