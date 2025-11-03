package com.millennialai.monitor.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.google.accompanist.swiperefresh.SwipeRefresh
import com.google.accompanist.swiperefresh.rememberSwipeRefreshState
import com.millennialai.monitor.ui.viewmodel.MonitoringViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun IntensiveMonitoringScreen(viewModel: MonitoringViewModel) {
    var selectedTab by remember { mutableStateOf(0) }
    val tabs = listOf("Health", "Metrics", "Diagnostics", "Tests", "Flow", "Performance", "Logs")

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MillennialAI Monitor") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(modifier = Modifier.padding(padding)) {
            TabRow(selectedTabIndex = selectedTab) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title) }
                    )
                }
            }

            when (selectedTab) {
                0 -> GenericTab(viewModel, "health") { viewModel.refreshHealth() }
                1 -> GenericTab(viewModel, "metrics") { viewModel.refreshMetrics() }
                2 -> GenericTab(viewModel, "diagnostics") { viewModel.refreshDiagnostics() }
                3 -> GenericTab(viewModel, "test-results") { viewModel.refreshTestResults() }
                4 -> GenericTab(viewModel, "injection-flow") { viewModel.refreshInjectionFlow() }
                5 -> GenericTab(viewModel, "performance") { viewModel.refreshPerformance() }
                6 -> GenericTab(viewModel, "logs") { viewModel.refreshLogs() }
            }
        }
    }
}

@Composable
fun GenericTab(
    viewModel: MonitoringViewModel,
    dataKey: String,
    onRefresh: () -> Unit
) {
    val refreshingMap by viewModel.isRefreshing.observeAsState(emptyMap())
    val isRefreshing = refreshingMap[dataKey] == true
    
    LaunchedEffect(Unit) { onRefresh() }
    
    SwipeRefresh(
        state = rememberSwipeRefreshState(isRefreshing),
        onRefresh = onRefresh
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Data from /$dataKey endpoint",
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
            }
        }
    }
}
