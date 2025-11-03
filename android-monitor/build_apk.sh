#!/bin/bash
# Build Android APK for MillennialAi Monitor

echo "🔨 Building MillennialAi Monitor APK for Galaxy S25..."

cd "$(dirname "$0")"

# Check if Android SDK is available
if ! command -v gradle &> /dev/null; then
    echo "❌ Gradle not found. Please install Android Studio or Gradle."
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
./gradlew clean

# Build release APK
echo "📦 Building release APK..."
./gradlew assembleRelease

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "📱 APK location:"
    echo "   app/build/outputs/apk/release/app-release.apk"
    echo ""
    echo "📲 To install on Galaxy S25:"
    echo "   adb install app/build/outputs/apk/release/app-release.apk"
    echo ""
    echo "   Or transfer the APK to your phone and install manually."
else
    echo "❌ Build failed!"
    exit 1
fi
