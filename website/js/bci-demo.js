/**
 * Real-time BCI Classification Demo
 * Simulates brain activity, EEG data, and motor imagery classification
 */

class BCIDemo {
    constructor() {
        this.isRunning = false;
        this.sessionStartTime = null;
        this.trialsProcessed = 0;
        this.sessionTime = 0;
        this.accuracy = 0;
        this.confidence = 0;
        this.currentPrediction = '-';
        this.updateInterval = null;
        this.classificationHistory = [];

        // Brain activity simulation
        this.leftPower = 0;
        this.rightPower = 0;
        this.eegData = [];
        this.waveformData = [];
        this.canvas = null;
        this.ctx = null;

        this.init();
    }

    init() {
        // Get DOM elements
        this.startBtn = document.getElementById('start-demo');
        this.stopBtn = document.getElementById('stop-demo');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');

        // Brain activity elements
        this.leftActivity = document.getElementById('left-activity');
        this.rightActivity = document.getElementById('right-activity');
        this.leftPowerEl = document.getElementById('left-power');
        this.rightPowerEl = document.getElementById('right-power');

        // Accuracy gauge elements
        this.accuracyFill = document.getElementById('accuracy-fill');
        this.accuracyValue = document.getElementById('accuracy-value');
        this.currentPredictionEl = document.getElementById('current-prediction');
        this.confidenceEl = document.getElementById('confidence');

        // EEG channel elements
        this.c3Fill = document.getElementById('c3-fill');
        this.czFill = document.getElementById('cz-fill');
        this.c4Fill = document.getElementById('c4-fill');
        this.c3Value = document.getElementById('c3-value');
        this.czValue = document.getElementById('cz-value');
        this.c4Value = document.getElementById('c4-value');

        // Metrics elements
        this.signalQuality = document.getElementById('signal-quality');
        this.trialsCount = document.getElementById('trials-count');
        this.sessionTimeEl = document.getElementById('session-time');
        this.updateRate = document.getElementById('update-rate');

        // History elements
        this.classificationHistoryEl = document.getElementById('classification-history');

        // Canvas setup
        this.canvas = document.getElementById('eeg-canvas');
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
        }

        // Event listeners
        this.startBtn.addEventListener('click', () => this.startSession());
        this.stopBtn.addEventListener('click', () => this.stopSession());

        // Initialize waveform data
        this.initializeWaveform();

        // Create brain activity dots
        this.createActivityDots();
    }

    createActivityDots() {
        // Create activity dots for brain visualization
        for (let i = 0; i < 12; i++) {
            const dot = document.createElement('div');
            dot.className = 'activity-dot';
            this.leftActivity.appendChild(dot);
        }

        for (let i = 0; i < 12; i++) {
            const dot = document.createElement('div');
            dot.className = 'activity-dot';
            this.rightActivity.appendChild(dot);
        }
    }

    initializeWaveform() {
        // Initialize waveform with zeros
        this.waveformData = new Array(100).fill(0);
        this.drawWaveform();
    }

    startSession() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.sessionStartTime = Date.now();
        this.trialsProcessed = 0;
        this.sessionTime = 0;
        this.classificationHistory = [];

        // Update UI
        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.statusIndicator.classList.add('active');
        this.statusText.textContent = 'Session Active';

        // Start real-time updates (10 Hz)
        this.updateInterval = setInterval(() => {
            this.updateSessionTime();
            this.simulateBrainActivity();
            this.simulateEEGData();
            this.simulateClassification();
            this.updateMetrics();
            this.drawWaveform();
        }, 100);

        showNotification('BCI Session Started!', 'success');
    }

    stopSession() {
        if (!this.isRunning) return;

        this.isRunning = false;
        clearInterval(this.updateInterval);

        // Update UI
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.statusIndicator.classList.remove('active');
        this.statusText.textContent = 'Session Stopped';

        showNotification('BCI Session Stopped', 'info');
    }

    updateSessionTime() {
        if (!this.sessionStartTime) return;

        const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;

        this.sessionTimeEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    simulateBrainActivity() {
        // Simulate motor imagery brain activity
        // Left vs Right hand motor imagery creates different power patterns
        const time = Date.now() / 1000;
        const motorActivity = Math.sin(time * 0.5) > 0 ? 'left' : 'right';

        if (motorActivity === 'left') {
            this.leftPower = 15 + Math.random() * 25;
            this.rightPower = 5 + Math.random() * 10;
        } else {
            this.leftPower = 5 + Math.random() * 10;
            this.rightPower = 15 + Math.random() * 25;
        }

        // Update power displays
        this.leftPowerEl.textContent = `${this.leftPower.toFixed(2)} µV`;
        this.rightPowerEl.textContent = `${this.rightPower.toFixed(2)} µV`;

        // Update brain visualization
        this.updateBrainVisualization();
    }

    updateBrainVisualization() {
        const leftDots = this.leftActivity.children;
        const rightDots = this.rightActivity.children;

        // Activate dots based on power levels
        const leftActiveCount = Math.floor((this.leftPower / 40) * leftDots.length);
        const rightActiveCount = Math.floor((this.rightPower / 40) * rightDots.length);

        for (let i = 0; i < leftDots.length; i++) {
            leftDots[i].classList.toggle('active', i < leftActiveCount);
        }

        for (let i = 0; i < rightDots.length; i++) {
            rightDots[i].classList.toggle('active', i < rightActiveCount);
        }

        // Update hemisphere highlighting
        const leftHemisphere = document.querySelector('.left-hemisphere');
        const rightHemisphere = document.querySelector('.right-hemisphere');

        leftHemisphere.classList.toggle('active', this.leftPower > this.rightPower * 1.2);
        rightHemisphere.classList.toggle('active', this.rightPower > this.leftPower * 1.2);
    }

    simulateEEGData() {
        // Simulate EEG channel data
        const c3Value = 8 + Math.random() * 12 + (this.leftPower / 10);
        const czValue = 6 + Math.random() * 8;
        const c4Value = 8 + Math.random() * 12 + (this.rightPower / 10);

        // Update channel bars
        this.c3Fill.style.height = `${(c3Value / 25) * 100}%`;
        this.czFill.style.height = `${(czValue / 25) * 100}%`;
        this.c4Fill.style.height = `${(c4Value / 25) * 100}%`;

        // Update channel values
        this.c3Value.textContent = c3Value.toFixed(2);
        this.czValue.textContent = czValue.toFixed(2);
        this.c4Value.textContent = c4Value.toFixed(2);

        // Update waveform data
        this.waveformData.shift();
        this.waveformData.push((c3Value + czValue + c4Value) / 3);
    }

    simulateClassification() {
        // Simulate classification based on brain activity
        const leftDominant = this.leftPower > this.rightPower * 1.1;
        const rightDominant = this.rightPower > this.leftPower * 1.1;

        if (leftDominant) {
            this.currentPrediction = 'Left Hand';
            this.confidence = 70 + Math.random() * 25;
        } else if (rightDominant) {
            this.currentPrediction = 'Right Hand';
            this.confidence = 70 + Math.random() * 25;
        } else {
            this.currentPrediction = 'Resting';
            this.confidence = 40 + Math.random() * 30;
        }

        // Add some noise to accuracy
        this.accuracy = 60 + Math.random() * 35;

        // Update UI
        this.currentPredictionEl.textContent = this.currentPrediction;
        this.confidenceEl.textContent = `${this.confidence.toFixed(1)}%`;

        // Update accuracy gauge
        const accuracyPercent = this.accuracy;
        const rotation = (accuracyPercent / 100) * 180;
        this.accuracyFill.style.clipPath = `polygon(50% 50%, 50% 0%, ${50 + 50 * Math.cos((rotation - 90) * Math.PI / 180)}% ${50 - 50 * Math.sin((rotation - 90) * Math.PI / 180)}%, 0% 100%, 100% 100%, 100% 0%)`;
        this.accuracyValue.textContent = `${accuracyPercent.toFixed(0)}%`;

        // Add to classification history occasionally
        if (Math.random() < 0.3) { // 30% chance each update
            this.addClassificationToHistory();
        }
    }

    addClassificationToHistory() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        const historyItem = {
            time: timeStr,
            prediction: this.currentPrediction,
            confidence: `${this.confidence.toFixed(1)}%`,
            accuracy: `${this.accuracy.toFixed(1)}%`
        };

        this.classificationHistory.unshift(historyItem);

        // Keep only last 20 items
        if (this.classificationHistory.length > 20) {
            this.classificationHistory.pop();
        }

        this.updateClassificationHistory();
    }

    updateClassificationHistory() {
        this.classificationHistoryEl.innerHTML = '';

        this.classificationHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <span class="history-time">${item.time}</span>
                <span class="history-prediction">${item.prediction}</span>
                <span class="history-confidence">${item.confidence}</span>
                <span class="history-accuracy">${item.accuracy}</span>
            `;
            this.classificationHistoryEl.appendChild(historyItem);
        });
    }

    updateMetrics() {
        // Update trials count (increment occasionally)
        if (Math.random() < 0.2) { // 20% chance each update
            this.trialsProcessed++;
            this.trialsCount.textContent = this.trialsProcessed;
        }

        // Update signal quality based on simulated data
        const avgPower = (this.leftPower + this.rightPower) / 2;
        if (avgPower > 25) {
            this.signalQuality.textContent = 'Excellent';
            this.signalQuality.style.color = '#10b981';
        } else if (avgPower > 15) {
            this.signalQuality.textContent = 'Good';
            this.signalQuality.style.color = '#f59e0b';
        } else {
            this.signalQuality.textContent = 'Poor';
            this.signalQuality.style.color = '#ef4444';
        }
    }

    drawWaveform() {
        if (!this.ctx) return;

        const canvas = this.canvas;
        const ctx = this.ctx;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;

        // Horizontal lines
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw waveform
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const dataLength = this.waveformData.length;
        const step = width / (dataLength - 1);

        this.waveformData.forEach((value, index) => {
            const x = index * step;
            const y = height - (value / 30) * height; // Scale to fit canvas

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Add glow effect
        ctx.shadowColor = '#2563eb';
        ctx.shadowBlur = 5;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
}

// Initialize BCI demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    const bciDemo = new BCIDemo();

    // Add some visual effects to the demo section
    const demoSection = document.querySelector('.bci-demo-section');
    if (demoSection) {
        // Add floating particles effect
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: absolute;
                width: 4px;
                height: 4px;
                background: rgba(59, 130, 246, 0.3);
                border-radius: 50%;
                top: ${Math.random() * 100}%;
                left: ${Math.random() * 100}%;
                animation: float ${3 + Math.random() * 4}s ease-in-out infinite;
                animation-delay: ${Math.random() * 3}s;
            `;
            demoSection.appendChild(particle);
        }
    }
});

// Add floating animation
if (!document.querySelector('#float-animation-style')) {
    const style = document.createElement('style');
    style.id = 'float-animation-style';
    style.textContent = `
        @keyframes float {
            0%, 100% {
                transform: translateY(0px) rotate(0deg);
                opacity: 0.3;
            }
            50% {
                transform: translateY(-20px) rotate(180deg);
                opacity: 0.8;
            }
        }
    `;
    document.head.appendChild(style);
}