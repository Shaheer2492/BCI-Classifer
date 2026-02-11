/**
 * ML-Enhanced Real-time BCI Classification Demo
 * Uses trained Gradient Boosting model for performance prediction
 */

class MLBCIDemo extends BCIDemo {
    constructor() {
        super();
        this.apiBaseUrl = 'http://127.0.0.1:5001/api';
        this.currentSubject = null;
        this.mlPrediction = null;
        this.useMLPredictions = false;
    }

    async init() {
        super.init();

        // Add ML-specific UI elements
        this.addMLControls();

        // Check if prediction server is available
        await this.checkServerHealth();
    }

    addMLControls() {
        // Add ML toggle button
        const controlPanel = document.querySelector('.control-panel');
        if (controlPanel) {
            const mlToggle = document.createElement('div');
            mlToggle.className = 'ml-toggle';
            mlToggle.innerHTML = `
                <label class="toggle-switch">
                    <input type="checkbox" id="ml-toggle" />
                    <span class="toggle-slider"></span>
                </label>
                <span class="toggle-label">Use ML Predictions</span>
            `;
            controlPanel.appendChild(mlToggle);

            // Add event listener
            document.getElementById('ml-toggle').addEventListener('change', (e) => {
                this.useMLPredictions = e.target.checked;
                if (this.useMLPredictions && !this.isRunning) {
                    this.loadNewSubject();
                }
            });
        }

        // Add ML info panel
        const demoGrid = document.querySelector('.demo-grid');
        if (demoGrid) {
            const mlInfoCard = document.createElement('div');
            mlInfoCard.className = 'demo-card ml-info';
            mlInfoCard.innerHTML = `
                <h3>🤖 ML Performance Prediction</h3>
                <div class="ml-info-container">
                    <div class="ml-status" id="ml-status">
                        <span class="status-dot"></span>
                        <span class="status-text">Checking server...</span>
                    </div>
                    <div class="ml-prediction-info" id="ml-prediction-info">
                        <div class="info-row">
                            <span class="info-label">Model:</span>
                            <span class="info-value">Random Forest</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Subject ID:</span>
                            <span class="info-value" id="ml-subject-id">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Predicted Accuracy:</span>
                            <span class="info-value" id="ml-predicted-acc">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Actual Accuracy:</span>
                            <span class="info-value" id="ml-actual-acc">-</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Prediction Error:</span>
                            <span class="info-value" id="ml-error">-</span>
                        </div>
                    </div>
                    <button class="btn btn-primary" id="load-subject-btn" style="margin-top: 1rem; width: 100%;">
                        Load New Subject
                    </button>
                </div>
            `;
            demoGrid.appendChild(mlInfoCard);

            // Add event listener for load subject button
            document.getElementById('load-subject-btn').addEventListener('click', () => {
                this.loadNewSubject();
            });
        }
    }

    async checkServerHealth() {
        const statusEl = document.getElementById('ml-status');
        if (!statusEl) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();

            if (data.status === 'healthy' && data.model_loaded) {
                statusEl.innerHTML = `
                    <span class="status-dot active"></span>
                    <span class="status-text">Server Online</span>
                `;
                return true;
            } else {
                throw new Error('Server not ready');
            }
        } catch (error) {
            statusEl.innerHTML = `
                <span class="status-dot offline"></span>
                <span class="status-text">Server Offline</span>
            `;
            console.error('ML server not available:', error);
            return false;
        }
    }

    async loadNewSubject() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/simulate_subject`);
            const data = await response.json();

            if (data.success) {
                this.currentSubject = data;
                this.updateMLInfo(data);
                showNotification(`Loaded Subject ${data.subject_id}`, 'success');
            }
        } catch (error) {
            console.error('Error loading subject:', error);
            showNotification('Failed to load subject data', 'warning');
        }
    }

    updateMLInfo(data) {
        document.getElementById('ml-subject-id').textContent = data.subject_id;
        document.getElementById('ml-predicted-acc').textContent =
            `${(data.predicted_accuracy * 100).toFixed(1)}%`;
        document.getElementById('ml-actual-acc').textContent =
            `${(data.actual_accuracy * 100).toFixed(1)}%`;

        const error = Math.abs(data.actual_accuracy - data.predicted_accuracy);
        document.getElementById('ml-error').textContent =
            `${(error * 100).toFixed(2)}%`;

        // Update the demo to use these values
        if (this.useMLPredictions) {
            this.accuracy = data.predicted_accuracy * 100;
            this.updateAccuracyGauge();
        }
    }

    updateAccuracyGauge() {
        const accuracyPercent = this.accuracy;
        const rotation = (accuracyPercent / 100) * 180;
        this.accuracyFill.style.clipPath = `polygon(50% 50%, 50% 0%, ${50 + 50 * Math.cos((rotation - 90) * Math.PI / 180)}% ${50 - 50 * Math.sin((rotation - 90) * Math.PI / 180)}%, 0% 100%, 100% 100%, 100% 0%)`;
        this.accuracyValue.textContent = `${accuracyPercent.toFixed(0)}%`;
    }

    simulateClassification() {
        // When using ML predictions, converge toward the ML-predicted value
        // instead of calling the base random function
        if (this.useMLPredictions && this.currentSubject) {
            // Determine prediction from brain activity
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

            this.currentPredictionEl.textContent = this.currentPrediction;
            this.confidenceEl.textContent = `${this.confidence.toFixed(1)}%`;

            // Converge toward ML-predicted accuracy over 15 trials
            const targetAccuracy = this.currentSubject.predicted_accuracy * 100;
            const maxTrials = 15;
            const convergenceRate = Math.min(this.trialsProcessed / maxTrials, 1.0);
            const noise = (1 - convergenceRate) * (Math.random() - 0.5) * 15;
            this.accuracy = 50 + (targetAccuracy - 50) * convergenceRate + noise;
            this.accuracy = Math.max(30, Math.min(100, this.accuracy));

            this.updateAccuracyGauge();

            if (Math.random() < 0.3) {
                this.addClassificationToHistory();
            }
        } else {
            // Non-ML mode: use base class convergence logic
            super.simulateClassification();
        }
    }
}

// Initialize ML-enhanced BCI demo when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if we should use ML demo or regular demo
    const urlParams = new URLSearchParams(window.location.search);
    const useML = urlParams.get('ml') === 'true';

    if (useML || window.location.pathname.includes('ml')) {
        const mlBciDemo = new MLBCIDemo();
        window.bciDemo = mlBciDemo;
    }
});

// Add CSS for ML-specific elements
if (!document.querySelector('#ml-demo-style')) {
    const style = document.createElement('style');
    style.id = 'ml-demo-style';
    style.textContent = `
        .ml-toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .toggle-slider {
            background-color: var(--accent-color);
        }

        input:checked + .toggle-slider:before {
            transform: translateX(26px);
        }

        .toggle-label {
            color: rgba(255, 255, 255, 0.9);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .ml-info-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .ml-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem;
            background: rgba(59, 130, 246, 0.1);
            border-radius: 8px;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #6b7280;
        }

        .status-dot.active {
            background: var(--accent-color);
            box-shadow: 0 0 10px var(--accent-color);
        }

        .status-dot.offline {
            background: #ef4444;
        }

        .ml-prediction-info {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            font-weight: 600;
            color: var(--text-secondary);
        }

        .info-value {
            font-weight: 700;
            color: var(--primary-color);
        }
    `;
    document.head.appendChild(style);
}
