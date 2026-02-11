/**
 * Neuralink Visualization Engine
 * Handles Canvas rendering for LFP (Local Field Potential) streams.
 */

/**
 * Neuralink Visualization Engine
 * Handles Canvas rendering for decomposed EEG frequency bands.
 */

class NeuralinkViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Configuration for 4 bands
        this.bands = [
            { name: 'Theta', color: '#9d00ff', freq: 6, amp: 0.5 },
            { name: 'Alpha', color: '#00ccff', freq: 10, amp: 0.5 },
            { name: 'Mu', color: '#00ff9d', freq: 12, amp: 0.5 },
            { name: 'Beta', color: '#ffcc00', freq: 20, amp: 0.5 }
        ];

        this.nChannels = 4;
        this.bufferLength = 400; // Horizontal resolution
        this.data = [];

        // Initialize empty buffers
        for (let i = 0; i < this.nChannels; i++) {
            this.data.push(new Array(this.bufferLength).fill(0));
        }

        // Animation loop
        this.isRunning = true;
        this.t = 0;
        this.animate = this.animate.bind(this);
        window.addEventListener('resize', () => this.resize());
        requestAnimationFrame(this.animate);
    }

    resize() {
        if (!this.canvas.parentElement) return;
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    /**
     * Update target amplitudes for each band based on subject features.
     * Normalized 0.0 - 1.0 expected.
     */
    updateAmplitudes(theta, alpha, mu, beta) {
        // Smoothly interpolate towards new targets (simple approach for now)
        this.bands[0].amp = theta;
        this.bands[1].amp = alpha;
        this.bands[2].amp = mu;
        this.bands[3].amp = beta;
    }

    // Generate next frame of data using sine waves + noise
    generateFrame() {
        this.t += 0.05;
        const newData = [];

        for (let i = 0; i < this.nChannels; i++) {
            const band = this.bands[i];

            // Base wave: sine at characteristic frequency
            let val = Math.sin(this.t * band.freq);

            // Add harmonics/noise for realism
            val += Math.sin(this.t * band.freq * 2.5) * 0.3;
            val += (Math.random() - 0.5) * 0.2;

            // Scale by current amplitude (feature-driven)
            val *= band.amp;

            newData.push(val);
        }

        // Shift and push
        for (let i = 0; i < this.nChannels; i++) {
            this.data[i].shift();
            this.data[i].push(newData[i]);
        }
    }

    draw() {
        if (!this.ctx) return;

        // Clear with fade
        this.ctx.fillStyle = 'rgba(5, 5, 5, 1)';
        this.ctx.fillRect(0, 0, this.width, this.height);

        this.generateFrame();

        const rowHeight = this.height / this.nChannels;
        const xStep = this.width / this.bufferLength;

        // Draw each channel
        for (let ch = 0; ch < this.nChannels; ch++) {
            const band = this.bands[ch];
            const yOffset = ch * rowHeight + (rowHeight / 2);

            this.ctx.beginPath();
            this.ctx.lineWidth = 2;
            this.ctx.strokeStyle = band.color;
            this.ctx.shadowBlur = 5;
            this.ctx.shadowColor = band.color;

            for (let i = 0; i < this.bufferLength; i++) {
                const x = i * xStep;
                // Scale for display
                const val = this.data[ch][i] * (rowHeight * 0.4);
                const y = yOffset + val;

                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.stroke();

            // Label
            this.ctx.fillStyle = band.color;
            this.ctx.font = '12px "Roboto Mono"';
            this.ctx.shadowBlur = 0;
            this.ctx.fillText(band.name + ' (' + band.freq + 'Hz)', 10, yOffset - rowHeight / 3);
        }

        // Draw scanning line
        const scanX = (Math.sin(this.t * 0.5) * 0.5 + 0.5) * this.width;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.fillRect(scanX, 0, 2, this.height);
    }

    animate() {
        if (!this.isRunning) return;
        this.draw();
        requestAnimationFrame(this.animate);
    }
}


/**
 * Score Gauge Visualization
 * Renders a 270-degree radial arc gauge for the BCI Control Quality score.
 */
class ScoreGaugeViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.currentScore = 0;
        this.targetScore = 0;
        this.tierColor = '#00ffcc';
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    resize() {
        if (!this.canvas.parentElement) return;
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    setScore(score, color) {
        this.targetScore = score;
        this.tierColor = color || '#00ffcc';
    }

    draw() {
        if (!this.ctx) return;
        // Smooth interpolation
        this.currentScore += (this.targetScore - this.currentScore) * 0.06;

        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h * 0.52;
        const radius = Math.min(cx, cy) * 0.78;

        // Arc range: 270 degrees (from 135deg to 405deg)
        const startAngle = Math.PI * 0.75;
        const endAngle = Math.PI * 2.25;
        const totalArc = endAngle - startAngle;

        // Track (background arc)
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, endAngle);
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = Math.max(8, radius * 0.12);
        ctx.lineCap = 'round';
        ctx.stroke();

        // Fill arc
        if (this.currentScore > 0.5) {
            const fillEnd = startAngle + (this.currentScore / 100) * totalArc;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, startAngle, fillEnd);
            ctx.strokeStyle = this.tierColor;
            ctx.lineWidth = Math.max(8, radius * 0.12);
            ctx.lineCap = 'round';
            ctx.shadowBlur = 15;
            ctx.shadowColor = this.tierColor;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        // Tick marks at 25, 50, 75
        const tickWidth = Math.max(6, radius * 0.06);
        for (const pct of [25, 50, 75]) {
            const tickAngle = startAngle + (pct / 100) * totalArc;
            const innerR = radius - tickWidth;
            const outerR = radius + tickWidth;
            ctx.beginPath();
            ctx.moveTo(cx + Math.cos(tickAngle) * innerR, cy + Math.sin(tickAngle) * innerR);
            ctx.lineTo(cx + Math.cos(tickAngle) * outerR, cy + Math.sin(tickAngle) * outerR);
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    animate() {
        this.draw();
        requestAnimationFrame(this.animate);
    }
}


/**
 * Learning Curve Visualization
 * Renders a trial-by-trial running accuracy line chart.
 */
class LearningCurveViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.runningAcc = [];
        this.sessionAvg = 0;
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        if (!this.canvas || !this.canvas.parentElement) return;
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        if (this.runningAcc.length > 0) this.draw();
    }

    setData(runningAcc, sessionAvg) {
        this.runningAcc = runningAcc;
        this.sessionAvg = sessionAvg;
        this.draw();
    }

    draw() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;
        const pad = { top: 10, right: 10, bottom: 22, left: 32 };

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(5, 5, 5, 1)';
        ctx.fillRect(0, 0, w, h);

        if (this.runningAcc.length === 0) return;

        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        // Grid lines at 25%, 50%, 75%
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = 1;
        for (const yVal of [0.25, 0.5, 0.75]) {
            const y = pad.top + plotH * (1 - yVal);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(pad.left + plotW, y);
            ctx.stroke();
            // Label
            ctx.fillStyle = '#333';
            ctx.font = '9px "Roboto Mono"';
            ctx.textAlign = 'right';
            ctx.fillText(Math.round(yVal * 100) + '%', pad.left - 4, y + 3);
        }
        ctx.textAlign = 'left';

        // Session average dashed line
        const avgY = pad.top + plotH * (1 - this.sessionAvg);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(0, 255, 204, 0.25)';
        ctx.beginPath();
        ctx.moveTo(pad.left, avgY);
        ctx.lineTo(pad.left + plotW, avgY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Data line
        const xStep = plotW / (this.runningAcc.length - 1 || 1);
        ctx.beginPath();
        ctx.strokeStyle = '#00ffcc';
        ctx.lineWidth = 2;
        ctx.shadowBlur = 5;
        ctx.shadowColor = '#00ffcc';

        for (let i = 0; i < this.runningAcc.length; i++) {
            const x = pad.left + i * xStep;
            const y = pad.top + plotH * (1 - this.runningAcc[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // X-axis label
        ctx.fillStyle = '#333';
        ctx.font = '9px "Roboto Mono"';
        ctx.textAlign = 'center';
        ctx.fillText('TRIAL', pad.left + plotW / 2, h - 4);
        ctx.textAlign = 'left';
    }
}
