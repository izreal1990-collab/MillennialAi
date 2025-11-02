#!/bin/bash
# SonarQube Local Scan Script for MillennialAi
# This script runs SonarQube analysis locally for development

set -e

echo "🔍 Running SonarQube scan for MillennialAi..."

# Set Java 21 explicitly for SonarQube scanner
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

# Check if local sonar-scanner exists
SCANNER_DIR="$(dirname "$0")/../sonar-scanner-4.8.0.2856-linux"
if [ ! -d "$SCANNER_DIR" ]; then
    echo "❌ Local sonar-scanner not found at $SCANNER_DIR"
    echo "   Please download from: https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.8.0.2856-linux.zip"
    exit 1
fi

# Check for SONAR_TOKEN environment variable
if [ -z "$SONAR_TOKEN" ]; then
    echo "⚠️  SONAR_TOKEN not set. Set it with: export SONAR_TOKEN=your_token"
    echo "   Get your token from: https://sonarcloud.io/account/security/"
    echo "   Continuing with anonymous scan (limited features)..."
fi

# Run tests with coverage first
echo "🧪 Running tests with coverage..."
if python -c "import pytest" 2>/dev/null; then
    python -m pytest --cov=millennial_ai --cov-report=xml:coverage.xml --cov-report=term-missing -v
else
    echo "⚠️  pytest not available, skipping test coverage. Install with: pip install pytest pytest-cov"
fi

# Run SonarQube scan using local scanner
echo "📊 Running SonarQube analysis..."
cd "$(dirname "$0")/.."
if [ -n "$SONAR_TOKEN" ]; then
    "$SCANNER_DIR/bin/sonar-scanner" -Dsonar.login="$SONAR_TOKEN"
else
    "$SCANNER_DIR/bin/sonar-scanner"
fi

echo "✅ SonarQube scan completed!"
echo "📈 View results at: https://sonarcloud.io/dashboard?id=millennial-ai"