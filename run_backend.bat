@echo off
REM Start FastAPI Backend Server
 
echo ================================
echo Starting Backend Server
echo ================================
echo.
 
call .venv\Scripts\activate.bat
 
echo Starting FastAPI on http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
 
cd backend
python app.py
cd ..