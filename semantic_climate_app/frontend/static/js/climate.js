/**
 * Climate visualization and metrics display
 */

let metricsChart = null;
let chartData = {
    labels: [],
    deltaKappa: [],
    alpha: [],
    deltaH: []
};

// Mode descriptions (10 base modes including coherence-derived)
const modeDescriptions = {
    'Sycophantic': 'Minimal exploration — AI mirrors user without meaningful deviation.',
    'Contemplative': 'Dense sustained meaning-making — deep, focused exploration.',
    'Emergent': 'Complexity developing — ideas forming, not yet dialectical.',
    'Generative': 'Building new conceptual ground — high curvature, low branching.',
    'Dialectical': 'Productive tension — thesis-antithesis dynamic, challenge-response.',
    'Resonant': 'Co-emergent coupling — balanced exploration and coherence.',
    'Chaotic': 'Conceptual instability — high variance, fragmenting patterns.',
    'Dissociative': 'Incoherent coupling — no clear semantic relationship.',
    'Liminal': 'Edge exploration — high complexity with semantic continuity.',
    'Transitional': 'Moving between modes — pattern developing.'
};

// Trajectory descriptions
const trajectoryDescriptions = {
    'warming': 'Rising toward complexity',
    'cooling': 'Falling toward simplicity',
    'oscillating': 'Mixed signals — dialectical movement',
    'stable': 'Holding pattern'
};

// Risk level descriptions
const riskDescriptions = {
    'low': 'Healthy coupling dynamics',
    'moderate': 'Some patterns worth monitoring',
    'high': 'Epistemic risk — potential enclosure',
    'critical': 'Critical — minimal engagement detected'
};

// Coherence pattern descriptions
const coherenceDescriptions = {
    'breathing': 'Healthy rhythm — meaning moves naturally',
    'transitional': 'Pattern developing',
    'locked': 'Repetitive pattern — stuck in loop',
    'fragmented': 'Fragmented — low semantic continuity',
    'insufficient_data': 'Gathering data...'
};

// Earth-warm color palette for chart
const chartColors = {
    temperature: 'rgb(205, 110, 70)',      // burnt orange
    humidity: 'rgb(115, 155, 155)',         // soft teal
    pressure: 'rgb(130, 155, 130)',         // sage moss
    temperatureBg: 'rgba(205, 110, 70, 0.15)',
    humidityBg: 'rgba(115, 155, 155, 0.15)',
    pressureBg: 'rgba(130, 155, 130, 0.15)'
};

// Initialize chart
document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('metricsChart').getContext('2d');

    metricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: 'Δκ (Temperature)',
                    data: chartData.deltaKappa,
                    borderColor: chartColors.temperature,
                    backgroundColor: chartColors.temperatureBg,
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 2
                },
                {
                    label: 'ΔH (Humidity)',
                    data: chartData.deltaH,
                    borderColor: chartColors.humidity,
                    backgroundColor: chartColors.humidityBg,
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 2
                },
                {
                    label: 'α (Pressure)',
                    data: chartData.alpha,
                    borderColor: chartColors.pressure,
                    backgroundColor: chartColors.pressureBg,
                    tension: 0.4,
                    pointRadius: 3,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Metric Value',
                        color: '#666'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#666'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Turn Count',
                        color: '#666'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#666'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#888',
                        font: {
                            family: "'SF Mono', 'Fira Code', 'Consolas', monospace",
                            size: 10
                        },
                        padding: 12
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(18, 18, 26, 0.95)',
                    titleColor: '#aaa',
                    bodyColor: '#888',
                    borderColor: '#333',
                    borderWidth: 1
                }
            }
        }
    });
});

function updateClimate(metrics) {
    const { climate, mode, coupling_mode, metrics: raw, psi_vector, attractor_basin, affective_substrate } = metrics;

    // Update gauges
    updateGauge('temp', raw.delta_kappa, climate.temp_level);
    updateGauge('humidity', raw.delta_h, climate.humidity_level);
    updateGauge('pressure', raw.alpha, climate.pressure_level);

    // Update coupling mode (use enhanced data if available)
    if (coupling_mode) {
        updateCouplingModeEnhanced(coupling_mode);
    } else {
        updateCouplingMode(mode);
    }

    // Update vector Ψ (4D phase-space)
    if (psi_vector) {
        updatePsiVector(psi_vector);
    }

    // Update attractor basin
    if (attractor_basin) {
        updateAttractorBasin(attractor_basin);
    }

    // Update affective substrate
    if (affective_substrate) {
        updateAffectiveSubstrate(affective_substrate);
    }
}

function updateGauge(type, value, level) {
    // Update gauge fill
    const percentage = Math.min(value * 100, 100);
    document.getElementById(`${type}Gauge`).style.width = `${percentage}%`;

    // Update value display
    document.getElementById(`${type}Value`).textContent = value.toFixed(3);

    // Update level badge
    const levelBadge = document.getElementById(`${type}Level`);
    levelBadge.textContent = level;

    // Earth-warm color coding (non-evaluative)
    const levelColors = {
        'LOW': 'rgb(130, 155, 130)',       // sage moss
        'MEDIUM': 'rgb(195, 140, 95)',     // amber ochre
        'HIGH': 'rgb(205, 110, 70)',       // burnt orange
        'OPTIMAL': 'rgb(100, 140, 160)'    // slate blue
    };

    levelBadge.style.backgroundColor = levelColors[level] || '#1a1a24';
    levelBadge.style.borderColor = levelColors[level] || '#333';
}

function updateCouplingMode(mode) {
    const modeIcons = {
        'Resonant': 'ph-check-circle',
        'Sycophantic': 'ph-warning-circle',
        'Chaotic': 'ph-x-circle',
        'Dissociative': 'ph-minus-circle',
        'Contemplative': 'ph-brain',
        'Emergent': 'ph-plant',
        'Generative': 'ph-lightbulb',
        'Dialectical': 'ph-arrows-left-right'
    };

    const modeClasses = {
        'Resonant': 'mode-resonant',
        'Sycophantic': 'mode-sycophantic',
        'Chaotic': 'mode-chaotic',
        'Dissociative': 'mode-dissociative',
        'Contemplative': 'mode-contemplative',
        'Emergent': 'mode-emergent',
        'Generative': 'mode-generative',
        'Dialectical': 'mode-dialectical'
    };

    const modeElement = document.getElementById('couplingMode');
    modeElement.className = 'mode-indicator ' + (modeClasses[mode] || 'mode-waiting');

    const iconElement = modeElement.querySelector('.mode-icon');
    iconElement.className = 'mode-icon ph ' + (modeIcons[mode] || 'ph-circle');

    modeElement.querySelector('.mode-text').textContent = mode;

    // Update description
    document.getElementById('modeDescription').textContent = modeDescriptions[mode] || '';

    // Hide trajectory and risk elements if using legacy mode
    const trajectoryEl = document.getElementById('trajectoryIndicator');
    const riskEl = document.getElementById('riskIndicator');
    if (trajectoryEl) trajectoryEl.style.display = 'none';
    if (riskEl) riskEl.style.display = 'none';
}

function updateCouplingModeEnhanced(couplingMode) {
    const { mode, trajectory, compound_label, epistemic_risk, risk_factors, confidence, coherence } = couplingMode;

    const modeIcons = {
        'Resonant': 'ph-check-circle',
        'Sycophantic': 'ph-warning-circle',
        'Chaotic': 'ph-x-circle',
        'Dissociative': 'ph-minus-circle',
        'Contemplative': 'ph-brain',
        'Emergent': 'ph-plant',
        'Generative': 'ph-lightbulb',
        'Dialectical': 'ph-arrows-left-right',
        'Liminal': 'ph-circles-three',
        'Transitional': 'ph-arrows-clockwise'
    };

    const modeClasses = {
        'Resonant': 'mode-resonant',
        'Sycophantic': 'mode-sycophantic',
        'Chaotic': 'mode-chaotic',
        'Dissociative': 'mode-dissociative',
        'Contemplative': 'mode-contemplative',
        'Emergent': 'mode-emergent',
        'Generative': 'mode-generative',
        'Dialectical': 'mode-dialectical',
        'Liminal': 'mode-liminal',
        'Transitional': 'mode-transitional'
    };

    const trajectoryIcons = {
        'warming': 'ph-trend-up',
        'cooling': 'ph-trend-down',
        'oscillating': 'ph-wave-sine',
        'stable': 'ph-minus'
    };

    const riskClasses = {
        'low': 'risk-low',
        'moderate': 'risk-moderate',
        'high': 'risk-high',
        'critical': 'risk-critical'
    };

    // Update main mode indicator
    const modeElement = document.getElementById('couplingMode');
    modeElement.className = 'mode-indicator ' + (modeClasses[mode] || 'mode-waiting');

    const iconElement = modeElement.querySelector('.mode-icon');
    iconElement.className = 'mode-icon ph ' + (modeIcons[mode] || 'ph-circle');

    // Show compound label (e.g., "Contemplative-Warming")
    modeElement.querySelector('.mode-text').textContent = compound_label;

    // Update description with base mode description
    document.getElementById('modeDescription').textContent = modeDescriptions[mode] || '';

    // Update trajectory indicator
    const trajectoryEl = document.getElementById('trajectoryIndicator');
    if (trajectoryEl) {
        trajectoryEl.style.display = 'flex';
        const trajIcon = trajectoryEl.querySelector('.trajectory-icon');
        const trajText = trajectoryEl.querySelector('.trajectory-text');
        if (trajIcon) trajIcon.className = 'trajectory-icon ph ' + (trajectoryIcons[trajectory] || 'ph-minus');
        if (trajText) trajText.textContent = trajectoryDescriptions[trajectory] || trajectory;
    }

    // Update risk indicator
    const riskEl = document.getElementById('riskIndicator');
    if (riskEl) {
        riskEl.style.display = 'flex';
        riskEl.className = 'risk-indicator ' + (riskClasses[epistemic_risk] || 'risk-low');
        const riskText = riskEl.querySelector('.risk-text');
        if (riskText) {
            if (risk_factors && risk_factors.length > 0) {
                riskText.textContent = risk_factors.map(f => f.replace(/_/g, ' ')).join(', ');
            } else {
                riskText.textContent = riskDescriptions[epistemic_risk] || '';
            }
        }
        const riskLabel = riskEl.querySelector('.risk-label');
        if (riskLabel) riskLabel.textContent = epistemic_risk.toUpperCase();
    }

    // Update confidence if element exists
    const confEl = document.getElementById('modeConfidence');
    if (confEl) {
        confEl.textContent = `${(confidence * 100).toFixed(0)}%`;
    }

    // Update coherence indicator
    const coherenceEl = document.getElementById('coherenceIndicator');
    if (coherenceEl && coherence) {
        coherenceEl.style.display = 'flex';

        const coherenceClasses = {
            'breathing': 'coherence-breathing',
            'transitional': 'coherence-transitional',
            'locked': 'coherence-locked',
            'fragmented': 'coherence-fragmented',
            'insufficient_data': 'coherence-waiting'
        };

        coherenceEl.className = 'coherence-indicator ' + (coherenceClasses[coherence.pattern] || 'coherence-transitional');

        const coherenceLabel = coherenceEl.querySelector('.coherence-label');
        if (coherenceLabel) coherenceLabel.textContent = coherence.pattern.replace(/_/g, ' ').toUpperCase();

        const coherenceText = coherenceEl.querySelector('.coherence-text');
        if (coherenceText) coherenceText.textContent = coherenceDescriptions[coherence.pattern] || '';

        const coherenceScore = coherenceEl.querySelector('.coherence-score');
        if (coherenceScore) coherenceScore.textContent = `${(coherence.coherence_score * 100).toFixed(0)}%`;

        const coherenceAutocorr = coherenceEl.querySelector('.coherence-autocorr');
        if (coherenceAutocorr) coherenceAutocorr.textContent = `r=${coherence.autocorrelation.toFixed(2)}`;
    } else if (coherenceEl) {
        coherenceEl.style.display = 'none';
    }
}

// Update Vector Ψ (4D Phase-Space)
function updatePsiVector(psi_vector) {
    // Map values from [-1, 1] to [0, 100] for display
    // 0% = -1 (left), 50% = 0 (center), 100% = +1 (right)
    const mapToPercent = (value) => {
        if (value === null) return 50;  // Center for null
        return (value + 1) * 50;  // Map [-1,1] to [0,100]
    };

    // Update semantic
    const semanticPercent = mapToPercent(psi_vector.semantic);
    document.getElementById('psiSemanticBar').style.width = `${semanticPercent}%`;
    document.getElementById('psiSemanticValue').textContent =
        psi_vector.semantic !== null ? psi_vector.semantic.toFixed(3) : '--';

    // Update temporal
    const temporalPercent = mapToPercent(psi_vector.temporal);
    document.getElementById('psiTemporalBar').style.width = `${temporalPercent}%`;
    document.getElementById('psiTemporalValue').textContent =
        psi_vector.temporal !== null ? psi_vector.temporal.toFixed(3) : '--';

    // Update affective
    const affectivePercent = mapToPercent(psi_vector.affective);
    document.getElementById('psiAffectiveBar').style.width = `${affectivePercent}%`;
    document.getElementById('psiAffectiveValue').textContent =
        psi_vector.affective !== null ? psi_vector.affective.toFixed(3) : '--';

    // Update biosignal
    const biosignalPercent = mapToPercent(psi_vector.biosignal);
    document.getElementById('psiBiosignalBar').style.width = `${biosignalPercent}%`;
    document.getElementById('psiBiosignalValue').textContent =
        psi_vector.biosignal !== null ? psi_vector.biosignal.toFixed(3) : 'None';
}

// Update Attractor Basin
function updateAttractorBasin(attractor_basin) {
    document.getElementById('basinName').textContent = attractor_basin.name;
    document.getElementById('basinConfidence').textContent =
        `${(attractor_basin.confidence * 100).toFixed(1)}%`;
}

// Update Affective Substrate
function updateAffectiveSubstrate(affective_substrate) {
    const affectiveDiv = document.getElementById('affectiveSubstrate');
    affectiveDiv.style.display = 'block';

    // Update basic metrics
    document.getElementById('affHedging').textContent =
        affective_substrate.hedging_density.toFixed(4);
    document.getElementById('affVulnerability').textContent =
        affective_substrate.vulnerability_score.toFixed(4);
    document.getElementById('affConfidence').textContent =
        affective_substrate.confidence_variance.toFixed(4);

    // Update source indicator
    const sourceEl = document.getElementById('affSource');
    const source = affective_substrate.source || 'vader';
    if (sourceEl) {
        sourceEl.textContent = source.toUpperCase();
        sourceEl.className = 'affective-source source-' + source;
    }

    // Handle GoEmotions extended data (hybrid mode)
    const emotionExtended = document.getElementById('emotionExtended');
    if (source === 'hybrid' && emotionExtended) {
        emotionExtended.style.display = 'block';

        // Update epistemic trajectory
        const epistemicMean = document.getElementById('epistemicMean');
        if (epistemicMean && affective_substrate.epistemic_mean !== undefined) {
            const val = affective_substrate.epistemic_mean;
            epistemicMean.textContent = val.toFixed(2);
            epistemicMean.className = 'trajectory-mean ' + getTrajectoryClass(val);
        }

        // Render epistemic sparkline
        // epistemic_trajectory is {curiosity: [...], confusion: [...], realization: [...], surprise: [...]}
        // Compute per-turn max across all epistemic emotions
        if (affective_substrate.epistemic_trajectory) {
            const epistemicValues = computeEpistemicPerTurn(affective_substrate.epistemic_trajectory);
            renderSparkline('epistemicSparkline', epistemicValues, 'epistemic');
        }

        // Update safety trajectory
        const safetyMean = document.getElementById('safetyMean');
        if (safetyMean && affective_substrate.safety_mean !== undefined) {
            const val = affective_substrate.safety_mean;
            safetyMean.textContent = val.toFixed(2);
            safetyMean.className = 'trajectory-mean ' + getTrajectoryClass(val);
        }

        // Render safety sparkline
        // safety_trajectory has {safety_positive: [...], safety_negative: [...], safety_net: [...]}
        // Use safety_net which is already (positive - negative)
        if (affective_substrate.safety_trajectory && affective_substrate.safety_trajectory.safety_net) {
            renderSparkline('safetySparkline', affective_substrate.safety_trajectory.safety_net, 'safety');
        }

        // Render top emotions
        if (affective_substrate.top_emotions) {
            renderTopEmotions(affective_substrate.top_emotions);
        }
    } else if (emotionExtended) {
        emotionExtended.style.display = 'none';
    }
}

// Get CSS class based on trajectory value
function getTrajectoryClass(value) {
    if (value > 0.3) return 'trajectory-high';
    if (value > 0) return 'trajectory-positive';
    if (value < -0.3) return 'trajectory-low';
    if (value < 0) return 'trajectory-negative';
    return 'trajectory-neutral';
}

// Compute per-turn epistemic score from trajectory object
// Takes {curiosity: [...], confusion: [...], realization: [...], surprise: [...]}
// Returns array of per-turn max values
function computeEpistemicPerTurn(epistemicTrajectory) {
    const emotions = ['curiosity', 'confusion', 'realization', 'surprise'];
    const arrays = emotions.map(e => epistemicTrajectory[e] || []);

    if (arrays.length === 0 || arrays[0].length === 0) return [];

    const numTurns = arrays[0].length;
    const result = [];

    for (let i = 0; i < numTurns; i++) {
        // Take max across all epistemic emotions for this turn
        let maxVal = 0;
        for (const arr of arrays) {
            if (arr[i] !== undefined && arr[i] > maxVal) {
                maxVal = arr[i];
            }
        }
        result.push(maxVal);
    }

    return result;
}

// Render a simple sparkline using SVG
function renderSparkline(containerId, values, type) {
    const container = document.getElementById(containerId);
    if (!container || !values || values.length === 0) return;

    const width = container.offsetWidth || 120;
    const height = 24;
    const padding = 2;

    // Values are probabilities [0, 1] — map to [height-padding, padding]
    const minVal = 0;
    const maxVal = 1;
    const range = maxVal - minVal;

    const points = values.map((v, i) => {
        const x = padding + (i / Math.max(1, values.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((v - minVal) / range) * (height - 2 * padding);
        return `${x},${y}`;
    }).join(' ');

    // Baseline at bottom (y = 0)
    const baselineY = height - padding;

    const strokeColor = type === 'epistemic' ? 'rgb(180, 140, 200)' : 'rgb(130, 180, 160)';
    const fillColor = type === 'epistemic' ? 'rgba(180, 140, 200, 0.2)' : 'rgba(130, 180, 160, 0.2)';

    // Create polygon points for fill (close the path at the bottom)
    const firstX = padding;
    const lastX = padding + ((values.length - 1) / Math.max(1, values.length - 1)) * (width - 2 * padding);
    const fillPoints = `${firstX},${baselineY} ${points} ${lastX},${baselineY}`;

    container.innerHTML = `
        <svg width="${width}" height="${height}" class="sparkline-svg">
            <polygon points="${fillPoints}" fill="${fillColor}"/>
            <polyline points="${points}" fill="none" stroke="${strokeColor}" stroke-width="1.5"/>
        </svg>
    `;
}

// Render top emotions as tags
function renderTopEmotions(topEmotions) {
    const container = document.getElementById('topEmotionsTags');
    if (!container) return;

    // Emotion category colors
    const emotionColors = {
        // Epistemic
        curiosity: 'emotion-epistemic',
        confusion: 'emotion-epistemic',
        realization: 'emotion-epistemic',
        surprise: 'emotion-epistemic',
        // Safety positive
        caring: 'emotion-safety-pos',
        gratitude: 'emotion-safety-pos',
        love: 'emotion-safety-pos',
        admiration: 'emotion-safety-pos',
        approval: 'emotion-safety-pos',
        relief: 'emotion-safety-pos',
        // Safety negative
        nervousness: 'emotion-safety-neg',
        fear: 'emotion-safety-neg',
        embarrassment: 'emotion-safety-neg',
        grief: 'emotion-safety-neg',
        sadness: 'emotion-safety-neg',
        // Other positive
        joy: 'emotion-positive',
        amusement: 'emotion-positive',
        excitement: 'emotion-positive',
        optimism: 'emotion-positive',
        pride: 'emotion-positive',
        // Other negative
        anger: 'emotion-negative',
        annoyance: 'emotion-negative',
        disappointment: 'emotion-negative',
        disapproval: 'emotion-negative',
        disgust: 'emotion-negative',
        remorse: 'emotion-negative',
        // Neutral
        neutral: 'emotion-neutral',
        desire: 'emotion-neutral'
    };

    // Take top 5 emotions
    const displayEmotions = topEmotions.slice(0, 5);

    container.innerHTML = displayEmotions.map(([emotion, score]) => {
        const colorClass = emotionColors[emotion] || 'emotion-neutral';
        return `<span class="emotion-tag ${colorClass}" title="${score.toFixed(2)}">${emotion}</span>`;
    }).join('');
}

/**
 * Update biosignal display from EBS (1Hz real-time)
 * Called by websocket handler when biosignal message received
 */
function updateBiosignal(biosignalData) {
    if (!biosignalData) return;

    // Map coherence [0,1] to Ψ_biosignal [-1,1]
    const coherence = biosignalData.coherence || 0;
    const psi_biosignal = (coherence * 2) - 1;

    // Map [-1, 1] to [0, 100] for display
    const mapToPercent = (value) => {
        if (value === null) return 50;
        return (value + 1) * 50;
    };

    // Update Ψ_biosignal bar and value
    const biosignalPercent = mapToPercent(psi_biosignal);
    const psiBiosignalBar = document.getElementById('psiBiosignalBar');
    const psiBiosignalValue = document.getElementById('psiBiosignalValue');
    if (psiBiosignalBar) psiBiosignalBar.style.width = `${biosignalPercent}%`;
    if (psiBiosignalValue) psiBiosignalValue.textContent = psi_biosignal.toFixed(3);

    // Update EBS status panel
    const ebsConnection = document.getElementById('ebsConnectionStatus');
    if (ebsConnection) {
        ebsConnection.classList.remove('disconnected');
        ebsConnection.classList.add('connected');
    }

    // Update HR display
    const hrDisplay = document.getElementById('ebsHR');
    if (hrDisplay && biosignalData.hr) {
        hrDisplay.textContent = biosignalData.hr;
    }

    // Update phase label
    const phaseLabel = document.getElementById('ebsPhaseLabel');
    if (phaseLabel && biosignalData.phase_label) {
        phaseLabel.textContent = biosignalData.phase_label;
    }

    // Show start session button once EBS is streaming
    const startBtn = document.getElementById('startSessionBtn');
    if (startBtn && !window.sessionStarted) {
        startBtn.style.display = 'flex';
    }
}

/**
 * Initialize EBS status panel and start session button
 */
function initEBSPanel() {
    const startBtn = document.getElementById('startSessionBtn');
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (window.sessionStarted) return;

            window.sessionStarted = true;
            startBtn.classList.add('session-active');
            startBtn.innerHTML = '<span>Session Active</span>';

            // Send field event to mark session start
            if (window.ws && window.ws.readyState === WebSocket.OPEN) {
                window.ws.send(JSON.stringify({
                    type: 'field_event',
                    event: 'session_start',
                    note: 'Formal session start marked by user'
                }));
            }

            // Add system message to chat
            if (typeof addSystemMessage === 'function') {
                addSystemMessage('Session started. Biosignal coupling active.');
            }
        });
    }
}

// Note: initEBSPanel is called from app.js after WebSocket connects

function updateChart(metrics) {
    const turnCount = parseInt(document.getElementById('turnCount').textContent);

    chartData.labels.push(turnCount);
    chartData.deltaKappa.push(metrics.metrics.delta_kappa);
    chartData.alpha.push(metrics.metrics.alpha);
    chartData.deltaH.push(metrics.metrics.delta_h);

    // Keep only last 20 points
    if (chartData.labels.length > 20) {
        chartData.labels.shift();
        chartData.deltaKappa.shift();
        chartData.alpha.shift();
        chartData.deltaH.shift();
    }

    metricsChart.update();
}

function resetClimate() {
    // Reset gauges
    ['temp', 'humidity', 'pressure'].forEach(type => {
        document.getElementById(`${type}Gauge`).style.width = '0%';
        document.getElementById(`${type}Value`).textContent = '--';
        document.getElementById(`${type}Level`).textContent = '--';
        document.getElementById(`${type}Level`).style.backgroundColor = '#64748b';
    });

    // Reset mode
    const modeElement = document.getElementById('couplingMode');
    modeElement.className = 'mode-indicator mode-waiting';
    const iconElement = modeElement.querySelector('.mode-icon');
    iconElement.className = 'mode-icon ph ph-circle';
    modeElement.querySelector('.mode-text').textContent = 'Waiting for analysis...';
    document.getElementById('modeDescription').textContent = 'Chat for at least 10 turns to begin analysis';

    // Reset chart
    chartData.labels = [];
    chartData.deltaKappa = [];
    chartData.alpha = [];
    chartData.deltaH = [];
    metricsChart.update();

    // Reset status
    document.getElementById('analysisStatus').textContent = 'Waiting for data...';
    document.getElementById('analysisStatus').className = 'status-waiting';
}
