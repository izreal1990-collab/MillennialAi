#!/bin/bash
# Generate Android launcher icons in multiple resolutions

echo "🎨 Generating Android launcher icons for MillennialAi Monitor..."

ANDROID_RES="/home/jovan-blango/Desktop/MillennialAi/android-monitor/app/src/main/res"
LOGO_SVG="/home/jovan-blango/Desktop/MillennialAi/logo.svg"

# Create directories if they don't exist
mkdir -p "$ANDROID_RES/mipmap-mdpi"
mkdir -p "$ANDROID_RES/mipmap-hdpi"
mkdir -p "$ANDROID_RES/mipmap-xhdpi"
mkdir -p "$ANDROID_RES/mipmap-xxhdpi"
mkdir -p "$ANDROID_RES/mipmap-xxxhdpi"

# Generate different sizes using ImageMagick
echo "📱 Generating mdpi (48x48)..."
convert -background none -density 300 "$LOGO_SVG" -resize 48x48 "$ANDROID_RES/mipmap-mdpi/ic_launcher.png"

echo "📱 Generating hdpi (72x72)..."
convert -background none -density 300 "$LOGO_SVG" -resize 72x72 "$ANDROID_RES/mipmap-hdpi/ic_launcher.png"

echo "📱 Generating xhdpi (96x96)..."
convert -background none -density 300 "$LOGO_SVG" -resize 96x96 "$ANDROID_RES/mipmap-xhdpi/ic_launcher.png"

echo "📱 Generating xxhdpi (144x144)..."
convert -background none -density 300 "$LOGO_SVG" -resize 144x144 "$ANDROID_RES/mipmap-xxhdpi/ic_launcher.png"

echo "📱 Generating xxxhdpi (192x192)..."
convert -background none -density 300 "$LOGO_SVG" -resize 192x192 "$ANDROID_RES/mipmap-xxxhdpi/ic_launcher.png"

# Round icons
echo "🔵 Generating round icons..."
convert -background none -density 300 "$LOGO_SVG" -resize 48x48 "$ANDROID_RES/mipmap-mdpi/ic_launcher_round.png"
convert -background none -density 300 "$LOGO_SVG" -resize 72x72 "$ANDROID_RES/mipmap-hdpi/ic_launcher_round.png"
convert -background none -density 300 "$LOGO_SVG" -resize 96x96 "$ANDROID_RES/mipmap-xhdpi/ic_launcher_round.png"
convert -background none -density 300 "$LOGO_SVG" -resize 144x144 "$ANDROID_RES/mipmap-xxhdpi/ic_launcher_round.png"
convert -background none -density 300 "$LOGO_SVG" -resize 192x192 "$ANDROID_RES/mipmap-xxxhdpi/ic_launcher_round.png"

echo "✅ All launcher icons generated!"
echo ""
echo "Generated icon resolutions:"
echo "  mdpi:    48x48"
echo "  hdpi:    72x72"
echo "  xhdpi:   96x96"
echo "  xxhdpi:  144x144"
echo "  xxxhdpi: 192x192"
echo ""
echo "Icons are in: $ANDROID_RES/mipmap-*/"
