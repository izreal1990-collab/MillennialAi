package com.millennialai.monitor.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.millennialai.monitor.data.HealthResponse
import com.millennialai.monitor.data.MetricsResponse
import com.millennialai.monitor.data.MonitorRepository
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class MonitoringUiState(
    val isLoading: Boolean = false,
    val health: HealthResponse? = null,
    val metrics: MetricsResponse? = null,
    val error: String? = null,
    val lastUpdated: Long = System.currentTimeMillis()
)

class MonitoringViewModel : ViewModel() {
    private val repository = MonitorRepository()
    private val _uiState = MutableStateFlow(MonitoringUiState())
    val uiState: StateFlow<MonitoringUiState> = _uiState.asStateFlow()
    
    private var monitoringJob: Job? = null
    
    fun startMonitoring() {
        monitoringJob?.cancel()
        monitoringJob = viewModelScope.launch {
            while (true) {
                fetchData()
                delay(10000) // Update every 10 seconds
            }
        }
    }
    
    fun refresh() {
        viewModelScope.launch {
            fetchData()
        }
    }
    
    private suspend fun fetchData() {
        _uiState.value = _uiState.value.copy(isLoading = true, error = null)
        
        val healthResult = repository.getHealth()
        val metricsResult = repository.getMetrics()
        
        _uiState.value = _uiState.value.copy(
            isLoading = false,
            health = healthResult.getOrNull(),
            metrics = metricsResult.getOrNull(),
            error = when {
                healthResult.isFailure && metricsResult.isFailure -> 
                    "Failed to fetch data: ${healthResult.exceptionOrNull()?.message}"
                healthResult.isFailure -> "Health check failed"
                metricsResult.isFailure -> "Metrics fetch failed"
                else -> null
            },
            lastUpdated = System.currentTimeMillis()
        )
    }
    
    override fun onCleared() {
        super.onCleared()
        monitoringJob?.cancel()
    }
}
