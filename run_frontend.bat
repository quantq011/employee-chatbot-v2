@echo off
REM Start Streamlit Frontend
 
echo ================================
echo Starting Frontend UI
echo ================================
echo.
 
call .venv\Scripts\activate.bat
 
echo Starting Streamlit on http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
 
cd frontend
streamlit run app.py
cd ..