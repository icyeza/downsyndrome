@echo off
REM Quick Start Script for Down Syndrome Classification Project
REM Windows version

echo.
echo ========================================================================
echo   DOWN SYNDROME CLASSIFICATION SYSTEM - QUICK START
echo ========================================================================
echo.

REM Check Python version
echo 2.14 Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo.

REM Check virtual environment
echo 2.14 Checking for virtual environment...
if "%VIRTUAL_ENV%"=="" (
    echo Warning: Virtual environment not detected!
    echo Run: python -m venv venv ^&^& venv\Scripts\activate
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)
echo.

REM Install dependencies
echo 2.14 Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Create directories
echo 2.14 Creating project directories...
if not exist "models" mkdir models
if not exist "data\train" mkdir data\train
if not exist "data\test" mkdir data\test
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs
echo Directories created!
echo.

echo ========================================================================
echo   QUICK START OPTIONS
echo ========================================================================
echo.
echo 1. Train the model:
echo    jupyter notebook notebook\down_syndrome.ipynb
echo.
echo 2. Start the API:
echo    cd api ^&^& python app.py
echo.
echo 3. Open the dashboard:
echo    cd ui ^&^& python -m http.server 8000
echo    Then open http://localhost:8000 in your browser
echo.
echo 4. Run load tests:
echo    locust -f tests\locustfile.py -H http://localhost:5000
echo.
echo 5. Deploy with Docker:
echo    docker-compose up -d
echo.
echo ========================================================================
echo   Setup complete! 3
echo ========================================================================
echo.
pause
