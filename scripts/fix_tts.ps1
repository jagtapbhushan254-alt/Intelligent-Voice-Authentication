#!/usr/bin/env pwsh
# Quick TTS Fix Script
# Run this to fix TTS issues

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  SecureX-Assist TTS Quick Fix" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Virtual environment not activated!" -ForegroundColor Red
    Write-Host "Please run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] Installing TTS dependencies..." -ForegroundColor Green
pip install --upgrade pyttsx3 pywin32 comtypes

Write-Host "`n[2/4] Testing pyttsx3..." -ForegroundColor Green
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('TTS test successful'); engine.runAndWait(); print('✅ pyttsx3 works')"

Write-Host "`n[3/4] Checking audio devices..." -ForegroundColor Green
python -c "import sounddevice as sd; print(f'Default output: {sd.query_devices(kind=''output'')[''name'']}'); print('✅ Audio device available')"

Write-Host "`n[4/4] Running full TTS test..." -ForegroundColor Green
python test_tts.py

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  TTS Fix Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run the main application: python main.py" -ForegroundColor White
Write-Host "2. Test voice commands" -ForegroundColor White
Write-Host "3. Check console output for TTS messages" -ForegroundColor White
