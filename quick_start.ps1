# 🚀 SecureX-Assist Quick Start
# Automated setup script for Windows PowerShell

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 58) -ForegroundColor Green
Write-Host "🔐 SECUREX-ASSIST - Voice Biometric Authentication" -ForegroundColor Cyan
Write-Host "   Quick Start Setup Script" -ForegroundColor White
Write-Host ("=" * 60) -ForegroundColor Green

# Check Python version
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.12+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n🔧 Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists" -ForegroundColor Yellow
    $response = Read-Host "   Delete and recreate? (y/n)"
    if ($response -eq "y") {
        Remove-Item -Recurse -Force venv
        Write-Host "   Deleted old environment" -ForegroundColor Gray
    }
}

if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "`n📚 Installing dependencies..." -ForegroundColor Yellow
Write-Host "   (This may take 5-10 minutes on first run)" -ForegroundColor Gray
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Dependency installation failed" -ForegroundColor Red
    exit 1
}

# Setup environment file
Write-Host "`n⚙️  Setting up environment..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Created .env file from template" -ForegroundColor Green
    Write-Host "   ⚠️  IMPORTANT: Edit .env and add your Hugging Face token!" -ForegroundColor Yellow
    Write-Host "   Get token from: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  .env file already exists" -ForegroundColor Yellow
}

# Initialize database
Write-Host "`n🗄️  Initializing database..." -ForegroundColor Yellow
$response = Read-Host "   Create test users? (y/n)"
if ($response -eq "y") {
    python init_db.py
}

# Summary
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Green
Write-Host ("=" * 58) -ForegroundColor Green
Write-Host "🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

Write-Host "`n📝 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Edit .env file and add your Hugging Face token" -ForegroundColor White
Write-Host "      Get token: https://huggingface.co/settings/tokens" -ForegroundColor Gray
Write-Host "   2. Accept model licenses:" -ForegroundColor White
Write-Host "      - https://huggingface.co/pyannote/embedding" -ForegroundColor Gray
Write-Host "      - https://huggingface.co/pyannote/segmentation" -ForegroundColor Gray
Write-Host "   3. Enroll your voice: python enroll_voice.py" -ForegroundColor White
Write-Host "   4. Launch app: python main.py" -ForegroundColor White

Write-Host "`n💡 Test Credentials (if you created test users):" -ForegroundColor Cyan
Write-Host "   Username: testuser" -ForegroundColor White
Write-Host "   Password: test123" -ForegroundColor White

Write-Host "`n🆘 Need Help?" -ForegroundColor Cyan
Write-Host "   - Read SETUP_GUIDE.md for detailed instructions" -ForegroundColor White
Write-Host "   - Check DOCUMENTATION.md for complete reference" -ForegroundColor White
Write-Host "   - Review securex.log for error messages" -ForegroundColor White

Write-Host "`n🚀 Ready to start? Run: python main.py`n" -ForegroundColor Green
