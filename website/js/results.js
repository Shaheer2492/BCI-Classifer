/**
 * Results page functionality
 * Loads and displays ground truth label data
 */

let resultsData = null;

// Load results from JSON file
async function loadResults() {
    const statusDiv = document.getElementById('phase1-status');

    try {
        statusDiv.innerHTML = '<div class="loading"></div> Loading results...';
        statusDiv.className = 'alert alert-info';

        // Attempt to load the results file
        const response = await fetch('../results/ground_truth_labels.json');

        if (!response.ok) {
            throw new Error('Results file not found. Please run Phase 1 script first.');
        }

        resultsData = await response.json();

        // Update status
        statusDiv.innerHTML = '<strong>Success!</strong> Results loaded successfully.';
        statusDiv.className = 'alert alert-success';

        // Display data
        displaySummaryStats();
        displaySubjectTable();

        showNotification('Results loaded successfully!', 'success');

    } catch (error) {
        console.error('Error loading results:', error);
        statusDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
        statusDiv.className = 'alert alert-warning';
        showNotification('Failed to load results', 'warning');
    }
}

// Display summary statistics
function displaySummaryStats() {
    if (!resultsData || !resultsData.summary) {
        return;
    }

    const summary = resultsData.summary;

    // Show summary section
    document.getElementById('summary-stats').style.display = 'block';

    // Update stat cards
    document.getElementById('total-subjects').textContent =
        `${summary.successful_subjects}/${summary.total_subjects}`;

    document.getElementById('mean-accuracy').textContent =
        formatPercentage(summary.mean_accuracy, 1);

    document.getElementById('median-accuracy').textContent =
        formatPercentage(summary.median_accuracy, 1);

    document.getElementById('accuracy-range').textContent =
        `${formatPercentage(summary.min_accuracy, 1)} - ${formatPercentage(summary.max_accuracy, 1)}`;
}

// Display subject performance table
function displaySubjectTable() {
    if (!resultsData || !resultsData.subjects) {
        return;
    }

    // Show table container
    document.getElementById('subject-table-container').style.display = 'block';

    const tbody = document.getElementById('subject-table-body');
    tbody.innerHTML = ''; // Clear existing rows

    // Sort subjects by ID
    const subjects = resultsData.subjects.sort((a, b) => a.subject_id - b.subject_id);

    // Create table rows
    subjects.forEach(subject => {
        const row = document.createElement('tr');

        // Apply color coding based on accuracy
        if (subject.success && subject.accuracy) {
            if (subject.accuracy >= 0.7) {
                row.style.backgroundColor = '#d1fae5'; // Green for high performers
            } else if (subject.accuracy < 0.5) {
                row.style.backgroundColor = '#fee2e2'; // Red for low performers
            }
        }

        row.innerHTML = `
            <td>${subject.subject_id}</td>
            <td>${subject.success ? formatPercentage(subject.accuracy, 1) : '-'}</td>
            <td>${subject.success ? formatPercentage(subject.accuracy_std, 1) : '-'}</td>
            <td>${subject.n_trials || '-'}</td>
            <td>${subject.n_channels || '-'}</td>
            <td>
                <span style="color: ${subject.success ? 'var(--accent-color)' : '#ef4444'}; font-weight: 600;">
                    ${subject.success ? '✓ Success' : '✗ Failed'}
                </span>
            </td>
        `;

        // Add click handler for details
        row.style.cursor = 'pointer';
        row.addEventListener('click', () => showSubjectDetails(subject));

        tbody.appendChild(row);
    });

    // Add performance distribution chart
    createDistributionChart();
}

// Show detailed information for a subject
function showSubjectDetails(subject) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;

    const content = document.createElement('div');
    content.style.cssText = `
        background: white;
        padding: 2rem;
        border-radius: 8px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    `;

    if (subject.success) {
        content.innerHTML = `
            <h2 style="margin-bottom: 1rem; color: var(--primary-color);">Subject ${subject.subject_id}</h2>

            <div style="margin: 1rem 0;">
                <h3 style="color: var(--text-primary);">Performance Metrics</h3>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                        <strong>Overall Accuracy:</strong> ${formatPercentage(subject.accuracy, 2)}
                    </li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                        <strong>Standard Deviation:</strong> ${formatPercentage(subject.accuracy_std, 2)}
                    </li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                        <strong>Number of Trials:</strong> ${subject.n_trials}
                    </li>
                    <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                        <strong>Number of Channels:</strong> ${subject.n_channels}
                    </li>
                </ul>
            </div>

            <div style="margin: 1rem 0;">
                <h3 style="color: var(--text-primary);">Fold Accuracies</h3>
                <ul style="list-style: none; padding: 0;">
                    ${subject.fold_accuracies.map((acc, idx) => `
                        <li style="padding: 0.25rem 0;">
                            Fold ${idx + 1}: ${formatPercentage(acc, 2)}
                        </li>
                    `).join('')}
                </ul>
            </div>

            <div style="margin: 1rem 0;">
                <h3 style="color: var(--text-primary);">Class Accuracies</h3>
                <ul style="list-style: none; padding: 0;">
                    ${Object.entries(subject.class_accuracies).map(([cls, acc]) => `
                        <li style="padding: 0.25rem 0;">
                            Class ${cls}: ${formatPercentage(acc, 2)}
                        </li>
                    `).join('')}
                </ul>
            </div>

            <div style="margin: 1rem 0;">
                <h3 style="color: var(--text-primary);">Class Distribution</h3>
                <ul style="list-style: none; padding: 0;">
                    ${Object.entries(subject.class_distribution).map(([cls, count]) => `
                        <li style="padding: 0.25rem 0;">
                            Class ${cls}: ${count} trials
                        </li>
                    `).join('')}
                </ul>
            </div>

            <button class="btn btn-primary" style="margin-top: 1rem; width: 100%;" onclick="this.closest('div[style*=\"position: fixed\"]').remove()">
                Close
            </button>
        `;
    } else {
        content.innerHTML = `
            <h2 style="margin-bottom: 1rem; color: var(--primary-color);">Subject ${subject.subject_id}</h2>
            <div class="alert alert-warning">
                <strong>Processing Failed</strong><br>
                ${subject.error || 'Unknown error occurred'}
            </div>
            <button class="btn btn-primary" style="margin-top: 1rem; width: 100%;" onclick="this.closest('div[style*=\"position: fixed\"]').remove()">
                Close
            </button>
        `;
    }

    modal.appendChild(content);
    document.body.appendChild(modal);

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Create performance distribution chart (simple ASCII/text-based)
function createDistributionChart() {
    if (!resultsData || !resultsData.subjects) {
        return;
    }

    const chartDiv = document.getElementById('distribution-chart');

    // Get accuracies from successful subjects
    const accuracies = resultsData.subjects
        .filter(s => s.success)
        .map(s => s.accuracy);

    if (accuracies.length === 0) {
        chartDiv.innerHTML = '<p style="text-align: center;">No data to display</p>';
        return;
    }

    // Create histogram bins
    const bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const counts = new Array(bins.length - 1).fill(0);

    accuracies.forEach(acc => {
        for (let i = 0; i < bins.length - 1; i++) {
            if (acc >= bins[i] && acc < bins[i + 1]) {
                counts[i]++;
                break;
            }
        }
    });

    // Create visual representation
    const maxCount = Math.max(...counts);
    const barWidth = 100 / (bins.length - 1);

    let html = '<div style="display: flex; align-items: flex-end; height: 300px; gap: 10px; padding: 20px;">';

    counts.forEach((count, i) => {
        const height = (count / maxCount) * 100;
        const label = `${formatPercentage(bins[i], 0)}-${formatPercentage(bins[i + 1], 0)}`;

        html += `
            <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 5px;">${count}</div>
                <div style="
                    width: 100%;
                    height: ${height}%;
                    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                    border-radius: 4px 4px 0 0;
                    min-height: 20px;
                    display: flex;
                    align-items: flex-end;
                    justify-content: center;
                "></div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: var(--text-secondary);">${label}</div>
            </div>
        `;
    });

    html += '</div>';
    html += '<p style="text-align: center; color: var(--text-secondary); margin-top: 1rem;">Accuracy Distribution (number of subjects per bin)</p>';

    chartDiv.innerHTML = html;
}

// Refresh results
function refreshResults() {
    loadResults();
}

// Format utility functions (ensure they're available)
function formatPercentage(num, decimals = 1) {
    return typeof num === 'number' ? (num * 100).toFixed(decimals) + '%' : num;
}
