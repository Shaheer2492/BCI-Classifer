/**
 * Neuralink Controller (Redesigned)
 * Story-driven interface: single score, signal quality, learning curve, benchmarking.
 */

// === DOM ELEMENTS ===
const els = {
    // Status bar
    statusDot: document.getElementById('connection-dot'),
    statusText: document.getElementById('connection-text'),
    signalStatus: document.getElementById('signal-status'),
    // Score panel
    scoreValue: document.getElementById('score-value'),
    scoreTier: document.getElementById('score-tier'),
    percentileText: document.getElementById('percentile-text'),
    scoreTip: document.getElementById('score-tip'),
    // Decoder panel
    intentText: document.getElementById('intent-text'),
    intentConf: document.getElementById('intent-conf'),
    barLeft: document.getElementById('bar-left'),
    barRight: document.getElementById('bar-right'),
    barRest: document.getElementById('bar-rest'),
    decoderRing: document.getElementById('decoder-ring-fill'),
    // Signal quality panel
    sigLeftFill: document.getElementById('sig-left-fill'),
    sigLeftStrength: document.getElementById('sig-left-strength'),
    sigRightFill: document.getElementById('sig-right-fill'),
    sigRightStrength: document.getElementById('sig-right-strength'),
    sigFocusFill: document.getElementById('sig-focus-fill'),
    sigFocusStrength: document.getElementById('sig-focus-strength'),
    signalTip: document.getElementById('signal-tip'),
    // Learning curve panel
    improvementPct: document.getElementById('improvement-pct'),
    bestStreak: document.getElementById('best-streak'),
    sessionAvg: document.getElementById('session-avg'),
    // Benchmark panel
    benchmarkUser: document.getElementById('benchmark-user'),
    benchmarkText: document.getElementById('benchmark-text'),
    milestonesContainer: document.getElementById('milestones-container'),
    // Controls
    consoleLog: document.getElementById('console-log'),
    btnSimulate: document.getElementById('btn-simulate'),
    subjectSelect: document.getElementById('subject-select')
};

// === STATE ===
let currentSubject = null;
let isConnected = false;
let updateInterval = null;
let populationAccuracies = null; // sorted array from /api/population_stats

// === COMPUTATION FUNCTIONS ===

function computeBCIScore(predicted, actual) {
    const blended = (actual * 0.7) + (predicted * 0.3);
    const MIN_ACC = 0.30;
    const MAX_ACC = 1.00;
    const raw = (blended - MIN_ACC) / (MAX_ACC - MIN_ACC);
    return Math.round(Math.max(0, Math.min(100, raw * 100)));
}

function getTier(score) {
    if (score >= 80) return { label: 'EXPERT CONTROL', color: '#00ffcc' };
    if (score >= 60) return { label: 'STRONG PERFORMER', color: '#00ccff' };
    if (score >= 40) return { label: 'MAKING PROGRESS', color: '#ffcc00' };
    return { label: 'LEARNING MODE', color: '#ff9933' };
}

function computePercentile(blendedAccuracy, sortedAccuracies) {
    if (sortedAccuracies && sortedAccuracies.length > 0) {
        let count = 0;
        for (const acc of sortedAccuracies) {
            if (acc < blendedAccuracy) count++;
        }
        return Math.round((count / sortedAccuracies.length) * 100);
    }
    // Fallback: normal CDF approximation
    const MEAN = 0.600, STD = 0.154;
    const z = (blendedAccuracy - MEAN) / STD;
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const d = 0.3989422804014327;
    const p = d * Math.exp(-z * z / 2) *
        (t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.330274)))));
    return z >= 0 ? Math.round((1 - p) * 100) : Math.round(p * 100);
}

// Signal quality from features (indices confirmed from early_trial_features.json)
function computeSignalQuality(features) {
    if (!features || features.length < 23) {
        return { left: 0.5, right: 0.5, focus: 0.5 };
    }

    // Left Motor (C3): ERD/ERS mu (idx 10) + beta (idx 11)
    const leftMag = Math.abs(features[10]) + Math.abs(features[11]) * 0.5;
    const left = Math.min(1, leftMag / 0.4);

    // Right Motor (C4): ERD/ERS mu (idx 14) + beta (idx 15)
    const rightMag = Math.abs(features[14]) + Math.abs(features[15]) * 0.5;
    const right = Math.min(1, rightMag / 0.4);

    // Focus Quality: SNR mean (idx 20), std (idx 21)
    const snrMean = features[20];
    const snrStd = features[21];
    const deviation = Math.abs(snrMean - 1.0);
    const consistencyFactor = 1.0 / (1.0 + snrStd * 2);
    const focus = Math.min(1, (deviation / 0.5) * consistencyFactor);

    return { left, right, focus };
}

function getStrengthLabel(value) {
    if (value >= 0.65) return { text: 'STRONG', color: '#00ffcc' };
    if (value >= 0.35) return { text: 'MODERATE', color: '#ffcc00' };
    return { text: 'WEAK', color: '#ff3333' };
}

function getSignalTip(left, right, focus) {
    const weakest = Math.min(left, right, focus);
    if (weakest >= 0.65) return 'All signals look great. Maintain steady focus.';
    if (weakest === focus && focus < 0.35) {
        return 'TIP: Maintain steady attention during imagery tasks for clearer signals.';
    }
    if (weakest === left && left < 0.35) {
        return 'TIP: Try vividly imagining left hand movements for stronger C3 activation.';
    }
    if (weakest === right && right < 0.35) {
        return 'TIP: Concentrate on right hand motor imagery to improve C4 signal.';
    }
    return 'Signals within normal range. Practice improves consistency.';
}

// Learning curve simulation from fold accuracies
function generateLearningCurve(foldAccuracies, nTrials) {
    if (!foldAccuracies || foldAccuracies.length === 0) {
        return { trialResults: [], runningAcc: [] };
    }

    const trialsPerBlock = Math.floor(nTrials / foldAccuracies.length);
    const trialResults = [];

    for (let foldIdx = 0; foldIdx < foldAccuracies.length; foldIdx++) {
        const foldAcc = foldAccuracies[foldIdx];
        const blockTrials = (foldIdx === foldAccuracies.length - 1)
            ? nTrials - trialResults.length  // last block gets remaining
            : trialsPerBlock;
        for (let t = 0; t < blockTrials; t++) {
            const learningBias = (t / blockTrials) * 0.05;
            const p = Math.min(1, foldAcc + learningBias);
            trialResults.push(Math.random() < p ? 1 : 0);
        }
    }

    // Running accuracy (window of 5)
    const windowSize = 5;
    const runningAcc = [];
    for (let i = 0; i < trialResults.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const window = trialResults.slice(start, i + 1);
        runningAcc.push(window.reduce((a, b) => a + b, 0) / window.length);
    }

    return { trialResults, runningAcc };
}

function computeLearningStats(trialResults) {
    if (!trialResults || trialResults.length === 0) {
        return { sessionAvg: 0, bestStreak: 0, improvement: 0 };
    }

    const sessionAvg = trialResults.reduce((a, b) => a + b, 0) / trialResults.length;

    let bestStreak = 0, currentStreak = 0;
    for (const r of trialResults) {
        if (r === 1) { currentStreak++; bestStreak = Math.max(bestStreak, currentStreak); }
        else { currentStreak = 0; }
    }

    const q = Math.max(1, Math.floor(trialResults.length / 4));
    const firstQ = trialResults.slice(0, q).reduce((a, b) => a + b, 0) / q;
    const lastQ = trialResults.slice(-q).reduce((a, b) => a + b, 0) / q;
    const improvement = Math.round((lastQ - firstQ) * 100);

    return { sessionAvg, bestStreak, improvement };
}

function getMotivationalMessage(score, improvement) {
    if (score >= 80) return 'Outstanding! You\'re among the top BCI performers.';
    if (score >= 60) return 'Impressive neural pattern differentiation.';
    if (score >= 40) {
        if (improvement > 0) return 'Solid progress. Your brain is adapting well.';
        return 'Developing BCI control. Practice strengthens the signal.';
    }
    if (improvement > 0) return 'Your brain is learning the patterns. Keep going!';
    return 'Every session builds new neural pathways. Consistency is key.';
}


// === CONTROLLER CLASS ===

class NeuralinkController {
    constructor() {
        this.lfpViz = new NeuralinkViz('lfp-canvas');
        this.scoreGauge = new ScoreGaugeViz('score-gauge-canvas');
        this.learningCurve = new LearningCurveViz('learning-curve-canvas');
        this.init();
    }

    init() {
        this.log('SYSTEM INITIALIZED');
        this.fetchPopulationStats();
        this.connect();

        if (els.btnSimulate) {
            els.btnSimulate.addEventListener('click', () => this.loadRandomSubject());
        }

        if (els.subjectSelect) {
            els.subjectSelect.addEventListener('change', (e) => {
                const id = e.target.value;
                if (id) this.loadSubject(id);
            });
            for (let i = 1; i <= 109; i++) {
                const opt = document.createElement('option');
                opt.value = i;
                opt.innerText = `Subject ${i}`;
                els.subjectSelect.appendChild(opt);
            }
        }
    }

    async fetchPopulationStats() {
        try {
            const res = await fetch('http://localhost:5001/api/population_stats');
            const data = await res.json();
            if (data.success) {
                populationAccuracies = data.accuracies;
                this.log('POPULATION DATA LOADED');
            }
        } catch (e) {
            console.warn('Population stats unavailable, using fallback percentile', e);
        }
    }

    connect() {
        this.log('CONNECTING TO LINK...');
        setTimeout(() => {
            isConnected = true;
            els.statusDot.style.backgroundColor = 'var(--accent-cyan)';
            els.statusText.innerText = 'CONNECTED';
            els.statusText.style.color = 'var(--accent-cyan)';
            this.log('LINK ESTABLISHED');
            this.loadSubject(1);
            this.startLoop();
        }, 1500);
    }

    async loadRandomSubject() {
        const randomId = Math.floor(Math.random() * 109) + 1;
        if (els.subjectSelect) els.subjectSelect.value = randomId;
        await this.loadSubject(randomId);
    }

    async loadSubject(id) {
        this.log(`FETCHING SUBJECT ${id} DATA...`);
        els.signalStatus.innerText = 'SYNCING...';

        try {
            const response = await fetch(`http://localhost:5001/api/subject/${id}`);
            const data = await response.json();

            if (data.success) {
                currentSubject = data;
                this.updateUI(data);
                this.log(`SUBJECT ${data.subject_id} SYNCED`);
                els.signalStatus.innerText = 'STRONG';
                els.signalStatus.className = 'value good';
            } else {
                this.log(`ERROR: ${data.error}`);
                els.signalStatus.innerText = 'NO SIGNAL';
                els.signalStatus.className = 'value bad';
            }
        } catch (e) {
            this.log('CONNECTION ERROR');
            console.error(e);
        }
    }

    updateUI(data) {
        // 1. BCI Control Quality Score
        const score = computeBCIScore(data.predicted_accuracy, data.actual_accuracy);
        const tier = getTier(score);
        const blended = (data.actual_accuracy * 0.7) + (data.predicted_accuracy * 0.3);
        const percentile = computePercentile(blended, populationAccuracies);

        if (this.scoreGauge) this.scoreGauge.setScore(score, tier.color);
        els.scoreValue.innerText = score;
        els.scoreTier.innerText = tier.label;
        els.scoreTier.style.color = tier.color;

        els.percentileText.innerText = `Better than ${percentile}% of users`;

        // 2. Signal Quality Bars
        const sq = computeSignalQuality(data.features);

        this.setSignalBar(els.sigLeftFill, els.sigLeftStrength, sq.left);
        this.setSignalBar(els.sigRightFill, els.sigRightStrength, sq.right);
        this.setSignalBar(els.sigFocusFill, els.sigFocusStrength, sq.focus);

        els.signalTip.innerText = getSignalTip(sq.left, sq.right, sq.focus);

        // 3. Mini LFP amplitudes (same mapping as before)
        if (data.features && data.features.length >= 6) {
            const minLog = -13, maxLog = -10;
            const norm = (v) => {
                if (v <= 0) return 0.1;
                const logVal = Math.log10(v);
                return Math.min(Math.max((logVal - minLog) / (maxLog - minLog), 0.1), 1.0);
            };
            this.lfpViz.updateAmplitudes(
                norm(data.features[0]),
                norm(data.features[1]),
                norm(data.features[2]),
                norm(data.features[3])
            );
        }

        // 4. Learning Curve
        const foldAcc = data.fold_accuracies || [];
        const nTrials = data.n_trials || 45;
        const { trialResults, runningAcc } = generateLearningCurve(foldAcc, nTrials);
        const stats = computeLearningStats(trialResults);

        if (this.learningCurve) this.learningCurve.setData(runningAcc, stats.sessionAvg);

        const impSign = stats.improvement >= 0 ? '+' : '';
        els.improvementPct.innerText = `${impSign}${stats.improvement}%`;
        els.improvementPct.style.color = stats.improvement >= 0 ? '#00ffcc' : '#ff3333';
        els.bestStreak.innerText = stats.bestStreak;
        els.sessionAvg.innerText = Math.round(stats.sessionAvg * 100) + '%';

        // 5. Population Benchmark
        const userPosition = ((blended - 0.31) / (0.98 - 0.31)) * 100;
        const clampedPos = Math.max(2, Math.min(98, userPosition));
        els.benchmarkUser.style.left = clampedPos + '%';
        els.benchmarkText.innerText =
            `Performing better than ${percentile}% of users`;

        // 6. Motivational message
        els.scoreTip.innerText = getMotivationalMessage(score, stats.improvement);

        // 7. Milestone check
        this.checkMilestones(score);
    }

    setSignalBar(fillEl, strengthEl, value) {
        const label = getStrengthLabel(value);
        fillEl.style.width = Math.round(value * 100) + '%';
        fillEl.style.backgroundColor = label.color;
        fillEl.style.boxShadow = `0 0 8px ${label.color}44`;
        strengthEl.innerText = label.text;
        strengthEl.style.color = label.color;
    }

    checkMilestones(score) {
        const milestones = [
            { threshold: 50, text: 'MILESTONE: Above Chance!' },
            { threshold: 70, text: 'MILESTONE: Advanced Control!' },
            { threshold: 90, text: 'MILESTONE: Elite Performer!' }
        ];

        for (const m of milestones) {
            if (score >= m.threshold) {
                this.showMilestone(m.text);
                break; // show highest applicable milestone
            }
        }
    }

    showMilestone(text) {
        if (!els.milestonesContainer) return;
        els.milestonesContainer.innerHTML = '';
        const toast = document.createElement('span');
        toast.className = 'milestone-toast';
        toast.innerText = text;
        els.milestonesContainer.appendChild(toast);
        // Remove after animation
        setTimeout(() => { toast.remove(); }, 3000);
    }

    startLoop() {
        if (updateInterval) clearInterval(updateInterval);
        updateInterval = setInterval(() => {
            if (!currentSubject) return;
            this.updateDecoderDisplay();
        }, 100);
    }

    updateDecoderDisplay() {
        const targetAcc = currentSubject.predicted_accuracy;
        const r = Math.random();
        let state = 'REST';
        let conf = 0;
        let leftProb = 0.1;
        let rightProb = 0.1;

        if (r > 0.6) {
            state = 'LEFT HAND';
            leftProb = 0.6 + (targetAcc * 0.3);
            rightProb = 0.1;
            conf = leftProb;
        } else if (r > 0.3) {
            state = 'RIGHT HAND';
            rightProb = 0.6 + (targetAcc * 0.3);
            leftProb = 0.1;
            conf = rightProb;
        } else {
            state = 'IDLE';
            conf = 0.8;
        }

        leftProb += (Math.random() - 0.5) * 0.1;
        rightProb += (Math.random() - 0.5) * 0.1;

        els.intentText.innerText = state;
        els.intentConf.innerText = (conf * 100).toFixed(1) + '%';
        els.intentText.style.color = state === 'IDLE' ? '#666' : 'var(--accent-cyan)';

        const setBar = (el, val) => {
            val = Math.max(0, Math.min(1, val));
            el.style.width = (val * 100) + '%';
        };

        setBar(els.barLeft, leftProb);
        setBar(els.barRight, rightProb);
        setBar(els.barRest, 1 - (leftProb + rightProb));

        const ringColor = state === 'IDLE' ? '#333' : 'var(--accent-cyan)';
        els.decoderRing.style.border = `2px solid ${ringColor}`;
    }

    log(msg) {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.innerHTML = `> ${msg}`;
        els.consoleLog.prepend(div);
        if (els.consoleLog.children.length > 20) {
            els.consoleLog.lastElementChild.remove();
        }
    }
}

// Start
window.neuralink = new NeuralinkController();
