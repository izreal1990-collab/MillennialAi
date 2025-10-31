#!/bin/bash

# Script to rename MillenialAi to MillennialAi throughout the project

echo "🔄 Fixing spelling from MillenialAi to MillennialAi..."

# First, let's rename the directory itself
echo "📁 Renaming directory..."
cd /home/jovan-blango/Desktop
if [ -d "MillenialAi" ]; then
    mv MillenialAi MillennialAi
    echo "✅ Directory renamed to MillennialAi"
fi

cd MillennialAi

# Find and replace in all files
echo "📝 Updating file contents..."

# Use find to get all text files and replace content
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -exec sed -i 's/MillenialAi/MillennialAi/g' {} \;

# Also fix GitHub URLs
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) -exec sed -i 's/izreal1990-collab\/MillenialAi/izreal1990-collab\/MillennialAi/g' {} \;

echo "✅ Content updated in all files"

# Update the package name in the directory structure if needed
if [ -d "millennial_ai" ]; then
    echo "📦 Package directory already correctly named"
else
    echo "⚠️  Package directory needs manual check"
fi

echo "🎉 Spelling correction complete!"
echo "📍 Project is now in: /home/jovan-blango/Desktop/MillennialAi"

# Show what was changed
echo ""
echo "🔍 Summary of changes:"
echo "   • Project name: MillenialAi → MillennialAi"
echo "   • Directory: MillenialAi/ → MillennialAi/"
echo "   • GitHub URLs: Updated to MillennialAi"
echo "   • All documentation: Updated"
echo "   • All code files: Updated"