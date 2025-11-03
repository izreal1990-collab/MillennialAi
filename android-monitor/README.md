<div align="center">

<img src="../logo.svg" alt="MillennialAi Logo" width="120"/>

# MillennialAi Monitor

**Android Performance Monitoring App**

</div>

Native Android monitoring app for tracking MillennialAi performance 24/7 on your Galaxy S25.

## Features

- **Real-time Status**: Live AI brain status and uptime tracking
- **Performance Metrics**: Brain load, memory usage, response times
- **24/7 Monitoring**: Background service with automatic updates every 10 seconds
- **Pull-to-Refresh**: Manual refresh capability
- **Beautiful UI**: Material Design 3 with MillennialAi purple brand colors
- **Charts & Graphs**: Visual performance tracking
- **Offline Support**: Graceful error handling when connection is lost

## Tech Stack

- **Kotlin** - Modern Android development
- **Jetpack Compose** - Declarative UI framework
- **Material Design 3** - Latest Material Design guidelines
- **Retrofit** - REST API client
- **Coroutines** - Async operations
- **ViewModel** - MVVM architecture
- **WorkManager** - Background tasks

## Build Instructions

### Prerequisites
- Android Studio Hedgehog (2023.1.1) or later
- JDK 17
- Android SDK 34
- Gradle 8.4+

### Build APK
```bash
cd android-monitor
./gradlew assembleRelease
```

The APK will be in: `app/build/outputs/apk/release/app-release.apk`

### Install on Galaxy S25
```bash
adb install app/build/outputs/apk/release/app-release.apk
```

Or transfer the APK to your phone and install manually.

## API Endpoints Used

- `GET /health` - System status and uptime
- `GET /metrics` - Performance metrics and statistics

## Permissions

- **INTERNET** - Connect to MillennialAi API
- **ACCESS_NETWORK_STATE** - Check connectivity
- **FOREGROUND_SERVICE** - Background monitoring
- **POST_NOTIFICATIONS** - Alert notifications
- **WAKE_LOCK** - Keep monitoring active

## Screenshots

### Main Dashboard
- 🟢 Live status indicator
- ⏱️ Uptime tracking
- 📊 Active conversations
- 💬 Total processed messages
- ⚡ Average response time
- ✅ Success rate
- 🧠 Brain load percentage
- 💾 Memory usage
- 📈 24-hour conversation count

### Pull-down Metrics
- Swipe down to refresh all data
- Auto-refresh every 10 seconds
- Last updated timestamp

## Configuration

API endpoint is configured in:
```kotlin
object ApiConfig {
    const val BASE_URL = "https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/"
}
```

To change the endpoint, edit `app/src/main/java/com/millennialai/monitor/data/ApiService.kt`

## License

Private - MillennialAi Project
