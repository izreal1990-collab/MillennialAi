#!/bin/bash

echo "🎨 Regenerating Android launcher icons with enhanced design..."

# Create mipmap directories if they don't exist
mkdir -p android-monitor/app/src/main/res/mipmap-mdpi
mkdir -p android-monitor/app/src/main/res/mipmap-hdpi
mkdir -p android-monitor/app/src/main/res/mipmap-xhdpi
mkdir -p android-monitor/app/src/main/res/mipmap-xxhdpi
mkdir -p android-monitor/app/src/main/res/mipmap-xxxhdpi

# Generate standard launcher icons
echo "📱 Generating mdpi (48x48)..."
convert -background none logo.svg -resize 48x48 android-monitor/app/src/main/res/mipmap-mdpi/ic_launcher.png

echo "📱 Generating hdpi (72x72)..."
convert -background none logo.svg -resize 72x72 android-monitor/app/src/main/res/mipmap-hdpi/ic_launcher.png

echo "📱 Generating xhdpi (96x96)..."
convert -background none logo.svg -resize 96x96 android-monitor/app/src/main/res/mipmap-xhdpi/ic_launcher.png

echo "📱 Generating xxhdpi (144x144)..."
convert -background none logo.svg -resize 144x144 android-monitor/app/src/main/res/mipmap-xxhdpi/ic_launcher.png

echo "📱 Generating xxxhdpi (192x192)..."
convert -background none logo.svg -resize 192x192 android-monitor/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png

# Generate round launcher icons
echo "🔵 Generating round icons..."
convert -background none logo.svg -resize 48x48 android-monitor/app/src/main/res/mipmap-mdpi/ic_launcher_round.png
convert -background none logo.svg -resize 72x72 android-monitor/app/src/main/res/mipmap-hdpi/ic_launcher_round.png
convert -background none logo.svg -resize 96x96 android-monitor/app/src/main/res/mipmap-xhdpi/ic_launcher_round.png
convert -background none logo.svg -resize 144x144 android-monitor/app/src/main/res/mipmap-xxhdpi/ic_launcher_round.png
convert -background none logo.svg -resize 192x192 android-monitor/app/src/main/res/mipmap-xxxhdpi/ic_launcher_round.png

echo "✅ All enhanced launcher icons generated!"
echo "📊 Total: 10 PNG files across 5 densities"
