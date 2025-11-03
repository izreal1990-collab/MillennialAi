# MillennialAi Branding Assets

## Logo Files

### Main Logo
- **File**: `logo.svg`
- **Size**: 400x400px
- **Format**: SVG (scalable vector)
- **Usage**: GitHub README, documentation, web

### Design Features
- **3D Effect**: Layered shadows and highlights
- **Colors**: 
  - Primary Purple: `#6366F1` (Indigo)
  - Secondary Purple: `#8B5CF6` (Violet)
  - Accent Pink: `#EC4899` (Pink)
  - Background: `#0F0F0F` (Dark)
- **Elements**:
  - Bold 3D letter "M" with gradient fill
  - "Ai" text overlapping the M (50% overlap)
  - Glow and shadow effects for depth
  - Circular dark background

## Android App Icons

### Launcher Icons
All Android launcher icons have been generated in multiple resolutions for different screen densities:

| Density | Size | Location |
|---------|------|----------|
| mdpi | 48x48 | `app/src/main/res/mipmap-mdpi/` |
| hdpi | 72x72 | `app/src/main/res/mipmap-hdpi/` |
| xhdpi | 96x96 | `app/src/main/res/mipmap-xhdpi/` |
| xxhdpi | 144x144 | `app/src/main/res/mipmap-xxhdpi/` |
| xxxhdpi | 192x192 | `app/src/main/res/mipmap-xxxhdpi/` |

### Files Generated
- `ic_launcher.png` - Standard square icon (all densities)
- `ic_launcher_round.png` - Round icon variant (all densities)
- `ic_launcher.xml` - Adaptive icon definition (API 26+)
- `ic_launcher_round.xml` - Adaptive round icon (API 26+)
- `ic_launcher_foreground.xml` - Foreground layer for adaptive icons
- `ic_launcher_background.xml` - Background color (#0F0F0F)

### Android Adaptive Icons
Starting with Android 8.0 (API 26), the app uses adaptive icons that can be:
- **Circular** - On stock Android devices
- **Rounded Square** - On Samsung devices
- **Squircle** - On Google Pixel devices
- **Teardrop** - On some OEM devices

The adaptive icon automatically adjusts to the device's icon shape while maintaining the MillennialAi branding.

## Regenerating Icons

To regenerate all Android launcher icons:

```bash
cd android-monitor
./generate_icons.sh
```

This requires ImageMagick to be installed.

## Brand Guidelines

### Color Palette
```
Primary:   #6366F1 (Indigo-500)
Secondary: #8B5CF6 (Violet-500)
Accent:    #EC4899 (Pink-500)
Dark:      #0F0F0F (Near Black)
Surface:   #212121 (Dark Gray)
```

### Typography
- **Primary Font**: Inter (Web/Android)
- **Fallback**: Arial, System Sans-serif
- **Weights**: Regular (400), Medium (500), Bold (700)

### Logo Usage
✅ **DO:**
- Maintain aspect ratio
- Use on dark backgrounds
- Keep minimum clearance around logo
- Use SVG for web/digital
- Use PNG for social media

❌ **DON'T:**
- Stretch or distort
- Change colors significantly
- Add drop shadows (already has 3D effect)
- Use on busy backgrounds
- Make smaller than 48x48px

## File Structure
```
MillennialAi/
├── logo.svg                              # Main SVG logo
├── android-monitor/
│   ├── generate_icons.sh                 # Icon generation script
│   └── app/src/main/res/
│       ├── mipmap-mdpi/                  # 48x48 icons
│       ├── mipmap-hdpi/                  # 72x72 icons
│       ├── mipmap-xhdpi/                 # 96x96 icons
│       ├── mipmap-xxhdpi/                # 144x144 icons
│       ├── mipmap-xxxhdpi/               # 192x192 icons
│       ├── mipmap-anydpi-v26/            # Adaptive icon definitions
│       └── values/                       # Background color
```

## Trademark Notice

The MillennialAi logo and brand assets are proprietary and confidential. Unauthorized use is prohibited.

© 2025 MillennialAi. All rights reserved.
