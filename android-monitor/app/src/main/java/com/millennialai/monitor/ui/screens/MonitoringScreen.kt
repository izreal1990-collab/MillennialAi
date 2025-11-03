package com.millennialai.monitor.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.swiperefresh.SwipeRefresh
import com.google.accompanist.swiperefresh.rememberSwipeRefreshState
import com.millennialai.monitor.ui.viewmodel.MonitoringViewModel
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MonitoringScreen(viewModel: MonitoringViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()
    val swipeRefreshState = rememberSwipeRefreshState(uiState.isLoading)
    
    LaunchedEffect(Unit) {
        viewModel.startMonitoring()
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "MillennialAi Monitor",
                            fontWeight = FontWeight.Bold,
                            fontSize = 20.sp
                        )
                        Text(
                            "24/7 Performance Tracking",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.7f)
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refresh() }) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Refresh"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary,
                    actionIconContentColor = MaterialTheme.colorScheme.onPrimary
                )
            )
        }
    ) { padding ->
        SwipeRefresh(
            state = swipeRefreshState,
            onRefresh = { viewModel.refresh() },
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .background(MaterialTheme.colorScheme.background),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Status Card
                item {
                    StatusCard(
                        status = uiState.health?.brain_status ?: "Unknown",
                        uptime = uiState.health?.uptime_seconds ?: 0.0,
                        isOnline = uiState.health?.status == "healthy"
                    )
                }
                
                // Quick Stats
                item {
                    QuickStatsGrid(
                        activeConversations = uiState.health?.active_conversations ?: 0,
                        totalProcessed = uiState.health?.total_processed ?: 0,
                        avgResponseTime = uiState.metrics?.avg_response_time_sec ?: 0.0,
                        successRate = uiState.metrics?.success_rate ?: 0.0
                    )
                }
                
                // ML Brain Architecture
                item {
                    MLBrainCard(
                        brainStatus = uiState.metrics?.brain_status ?: "unknown",
                        layersCount = uiState.metrics?.brain_layers_count ?: 0,
                        totalParameters = uiState.metrics?.brain_total_parameters ?: 0,
                        brainLoad = uiState.metrics?.brain_load_percentage ?: 0.0
                    )
                }
                
                // Learning System
                item {
                    LearningSystemCard(
                        learningActive = uiState.metrics?.learning_active ?: false,
                        queueSize = uiState.metrics?.learning_queue_size ?: 0,
                        samplesProcessed = uiState.metrics?.learning_samples_processed ?: 0,
                        learningProgress = uiState.metrics?.learning_progress_percentage ?: 0.0,
                        threshold = uiState.metrics?.learning_threshold ?: 10000
                    )
                }
                
                // Performance Metrics
                item {
                    PerformanceCard(
                        brainLoad = uiState.metrics?.brain_load_percentage ?: 0.0,
                        memoryUsage = uiState.metrics?.memory_usage_mb ?: 0.0,
                        memoryPercent = uiState.metrics?.memory_percent ?: 0.0,
                        cpuPercent = uiState.metrics?.cpu_percent ?: 0.0,
                        requestCount = uiState.metrics?.total_requests ?: 0,
                        conversations24h = uiState.metrics?.conversations_24h ?: 0
                    )
                }
                
                // Context Window Usage
                item {
                    ContextWindowCard(
                        maxTokens = uiState.metrics?.max_context_tokens ?: 4096,
                        avgUsagePercent = uiState.metrics?.avg_context_usage_percentage ?: 0.0,
                        avgConversationLength = uiState.metrics?.avg_conversation_length ?: 0.0
                    )
                }
                
                // Error Display
                uiState.error?.let { error ->
                    item {
                        ErrorCard(error = error)
                    }
                }
                
                // Last Updated
                item {
                    Text(
                        text = "Last updated: ${formatTimestamp(uiState.lastUpdated)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
}

@Composable
fun StatusCard(status: String, uptime: Double, isOnline: Boolean) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (isOnline) Color(0xFF1B5E20) else Color(0xFFB71C1C)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = if (isOnline) "🟢 ONLINE" else "🔴 OFFLINE",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Brain Status: $status",
                    fontSize = 16.sp,
                    color = Color.White.copy(alpha = 0.9f)
                )
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = "Uptime",
                    fontSize = 12.sp,
                    color = Color.White.copy(alpha = 0.7f)
                )
                Text(
                    text = formatUptime(uptime),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
        }
    }
}

@Composable
fun QuickStatsGrid(
    activeConversations: Int,
    totalProcessed: Int,
    avgResponseTime: Double,
    successRate: Double
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        StatCard(
            title = "Active",
            value = activeConversations.toString(),
            modifier = Modifier.weight(1f),
            gradient = Brush.linearGradient(
                colors = listOf(Color(0xFF6366F1), Color(0xFF8B5CF6))
            )
        )
        StatCard(
            title = "Processed",
            value = totalProcessed.toString(),
            modifier = Modifier.weight(1f),
            gradient = Brush.linearGradient(
                colors = listOf(Color(0xFF8B5CF6), Color(0xFFEC4899))
            )
        )
    }
    Spacer(modifier = Modifier.height(8.dp))
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        StatCard(
            title = "Avg Response",
            value = "${String.format("%.2f", avgResponseTime)}s",
            modifier = Modifier.weight(1f),
            gradient = Brush.linearGradient(
                colors = listOf(Color(0xFF10B981), Color(0xFF059669))
            )
        )
        StatCard(
            title = "Success Rate",
            value = "${String.format("%.1f", successRate)}%",
            modifier = Modifier.weight(1f),
            gradient = Brush.linearGradient(
                colors = listOf(Color(0xFF3B82F6), Color(0xFF2563EB))
            )
        )
    }
}

@Composable
fun StatCard(title: String, value: String, modifier: Modifier = Modifier, gradient: Brush) {
    Card(
        modifier = modifier.height(100.dp),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color.Transparent),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(gradient)
                .padding(12.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = title,
                    fontSize = 12.sp,
                    color = Color.White.copy(alpha = 0.8f),
                    fontWeight = FontWeight.Medium
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = value,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
        }
    }
}

@Composable
fun PerformanceCard(
    brainLoad: Double,
    memoryUsage: Double,
    memoryPercent: Double,
    cpuPercent: Double,
    requestCount: Int,
    conversations24h: Int
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "⚡ System Performance",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            MetricRow("CPU Usage", "${String.format("%.1f", cpuPercent)}%", cpuPercent)
            Spacer(modifier = Modifier.height(12.dp))
            MetricRow("Memory Usage", "${String.format("%.0f", memoryUsage)} MB (${String.format("%.1f", memoryPercent)}%)", memoryPercent)
            Spacer(modifier = Modifier.height(12.dp))
            MetricRow("Brain Load", "${String.format("%.1f", brainLoad)}%", brainLoad)
            
            Spacer(modifier = Modifier.height(12.dp))
            Divider(modifier = Modifier.padding(vertical = 8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Total Requests", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(requestCount.toString(), fontSize = 20.sp, fontWeight = FontWeight.Bold)
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("24h Conversations", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(conversations24h.toString(), fontSize = 20.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
fun MLBrainCard(
    brainStatus: String,
    layersCount: Int,
    totalParameters: Int,
    brainLoad: Double
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF8B5CF6).copy(alpha = 0.1f),
            contentColor = MaterialTheme.colorScheme.onSurface
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "🧠 ML Brain Architecture",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF8B5CF6)
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Status", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(
                        text = brainStatus.uppercase(),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (brainStatus == "ready") Color(0xFF10B981) else Color(0xFFEF4444)
                    )
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("Neural Layers", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(layersCount.toString(), fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFF8B5CF6))
                }
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Total Parameters", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(
                        text = formatNumber(totalParameters),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("Brain Load", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text("${String.format("%.1f", brainLoad)}%", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color(0xFF6366F1))
                }
            }
        }
    }
}

@Composable
fun LearningSystemCard(
    learningActive: Boolean,
    queueSize: Int,
    samplesProcessed: Int,
    learningProgress: Double,
    threshold: Int
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFEC4899).copy(alpha = 0.1f)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "📚 Continuous Learning",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFEC4899)
                )
                Text(
                    text = if (learningActive) "🟢 ACTIVE" else "⏸️ STANDBY",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (learningActive) Color(0xFF10B981) else Color(0xFFFBBF24)
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "Learning Progress to Threshold",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            Text(
                text = "$samplesProcessed / $threshold samples",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            LinearProgressIndicator(
                progress = (learningProgress / 100).toFloat().coerceIn(0f, 1f),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(12.dp),
                color = Color(0xFFEC4899),
                trackColor = MaterialTheme.colorScheme.surfaceVariant
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Queue Size", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(queueSize.toString(), fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFFEC4899))
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("Progress", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text("${String.format("%.1f", learningProgress)}%", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFFEC4899))
                }
            }
        }
    }
}

@Composable
fun ContextWindowCard(
    maxTokens: Int,
    avgUsagePercent: Double,
    avgConversationLength: Double
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF10B981).copy(alpha = 0.1f)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "💬 Context Window",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF10B981)
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Max Tokens", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text(formatNumber(maxTokens), fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFF10B981))
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("Avg Usage", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    Text("${String.format("%.1f", avgUsagePercent)}%", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFF10B981))
                }
            }
            
            Spacer(modifier = Modifier.height(12.dp))
            
            Column {
                Text("Avg Conversation Length", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                Text("${String.format("%.1f", avgConversationLength)} exchanges", fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
fun MetricRow(label: String, value: String, progress: Double) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(label, fontSize = 14.sp)
            Text(value, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        }
        Spacer(modifier = Modifier.height(4.dp))
        LinearProgressIndicator(
            progress = (progress / 100).toFloat().coerceIn(0f, 1f),
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
            color = when {
                progress > 80 -> Color(0xFFEF4444)
                progress > 50 -> Color(0xFFFBBF24)
                else -> Color(0xFF10B981)
            },
            trackColor = MaterialTheme.colorScheme.surfaceVariant
        )
    }
}

@Composable
fun ErrorCard(error: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFB71C1C).copy(alpha = 0.1f)
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("⚠️", fontSize = 24.sp)
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = error,
                color = Color(0xFFB71C1C),
                fontSize = 14.sp
            )
        }
    }
}

fun formatUptime(seconds: Double): String {
    val hours = (seconds / 3600).toInt()
    val minutes = ((seconds % 3600) / 60).toInt()
    return if (hours > 0) "${hours}h ${minutes}m" else "${minutes}m"
}

fun formatTimestamp(millis: Long): String {
    val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    return sdf.format(Date(millis))
}

fun formatNumber(num: Int): String {
    return when {
        num >= 1_000_000 -> String.format("%.1fM", num / 1_000_000.0)
        num >= 1_000 -> String.format("%.1fK", num / 1_000.0)
        else -> num.toString()
    }
}
