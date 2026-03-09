# BCI Performance Prediction — Presentation Script
*~3–4 minutes*

---

## Opening (30 sec)

"Today I'll present our BCI Performance Prediction system. The goal is simple: **predict how well someone will control a brain–computer interface using only their first 15 trials**—instead of running a full calibration. That means faster setup and early screening of who will be a strong BCI user."

---

## The Problem & Approach (45 sec)

"BCI calibration is slow. We typically need many trials to train a decoder and estimate performance. We asked: **Can we predict final accuracy from early trials alone?**

We use the PhysioNet Motor Imagery dataset—109 subjects, 64 EEG channels. Our pipeline has four phases: First, we compute ground truth—final accuracy from a full CSP+LDA decoder on all trials. Second, we extract neurophysiological features from only the first 15 trials—band power, CSP patterns, ERD/ERS, variability, and SNR. Third, we merge in additional features from our team—Daniel’s SMR and ERD metrics, Andrew’s entropy and complexity features, and resting-state predictors. That gives us 38 features, which we reduce to 12 using rigorous feature selection: Spearman correlation, permutation tests, and FDR correction."

---

## Models & Results (90 sec)

"We train two kinds of models.

**Regression:** We predict continuous accuracy. We compare four models—Random Forest, Ridge, SVM, and Gradient Boosting. Random Forest performs best: R² of 0.30, Pearson r of 0.58, RMSE of 0.13. An R² of 0.30 might sound low, but it’s realistic when predicting from only 15 trials. Earlier work showed R² around 0.95, but that used circular features—fold accuracies from the full dataset. Our approach avoids that.

**Classifier:** We also predict HIGH vs LOW performer using a threshold of 65% accuracy. Our Random Forest classifier achieves 79% leave-one-out accuracy. It’s better at identifying LOW performers—89% recall—which is useful for screening. We use class weighting to handle the imbalance: 73 LOW vs 26 HIGH subjects.

The 12 selected features span band power at C3 and C4, CSP class separability, mu desynchronization, SNR, Daniel’s RPL and SMR strength, and resting-state predictors. These were validated in our EDA as significant predictors of BCI performance."

---

## Impact & Wrap-Up (45 sec)

"What does this enable? **Faster calibration**—predict performance without full training. **User screening**—identify strong BCI users early. **Adaptive training**—adjust protocols based on predictions.

We’ve built an end-to-end pipeline: ground truth generation, early-trial feature extraction, merged features, model training, and a Flask API for real-time predictions. The system is production-ready and documented.

Thank you. Happy to take questions."

---

## Backup Slides / Q&A Notes

- **Why 15 trials?** Balance between early prediction and enough signal; avoids data leakage.
- **Why Random Forest?** Best R²; handles non-linearity; interpretable feature importance.
- **Tiering:** EXPERT (≥80%), PROFICIENT (≥65%), DEVELOPING (≥50%), EARLY (<50%).
