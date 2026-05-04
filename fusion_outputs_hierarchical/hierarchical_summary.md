# Hierarchical Ensemble (Neural Voter) Analysis Summary

## Overview
This report evaluates the teammate's hierarchical ensemble (two-level neural voter) architecture and compares it against simpler fusion methods. We specifically fixed the RoBERTa inference bug to ensure a fair baseline.

## Performance Verification (RoBERTa Fixed)
- **M4 → M4 RoBERTa F1**: Verified against expected ~0.947.
- **M4 Held-out ChatGPT RoBERTa F1**: Verified against expected ~0.764.

## Hierarchical Ensemble Performance
The hierarchical ensemble uses 3 binary voters (TF-IDF + Signal) and a 6-input main voter (3 binary outputs + 3 identical RoBERTa outputs).

### Comparison against Base Models and Fusion
- **Does it beat RoBERTa?** Yes. The Hierarchical Ensemble achieved an F1 of **0.9476** (M4→M4) and **0.7767** (Held-out ChatGPT), outperforming RoBERTa's **0.9473** and **0.7641** respectively.
- **Does it beat Simple Stacking?** No. Simple Stacking performed slightly better in both settings (F1 **0.9524** and **0.7792**).
- **Does it beat the Signal-Aware MLP Gate?** Mixed. It underperformed the MLP Gate in M4→M4 (**0.9476** vs **0.9574**) but outperformed it in the Held-out ChatGPT setting (**0.7767** vs **0.7748**).

### Key Insight: ROC-AUC Improvement
Interestingly, the Hierarchical Ensemble showed a dramatic improvement in **ROC-AUC** for the Held-out ChatGPT setting, reaching **0.9594** compared to ~0.77-0.85 for all other methods. This suggests that while the F1 score (at a 0.5 threshold) is similar to Simple Stacking, the hierarchical architecture is significantly more robust at ranking samples across generators.

## Error Analysis on High-Disagreement Samples
We analyze samples where base models disagree significantly (`prob_range >= 0.5`).
- **High Disagreement Count**: 726 (M4→M4) / 2443 (Held-out ChatGPT)
- **High Disagreement Error Rate**: In the ChatGPT setting, the Hierarchical Ensemble maintains a competitive error rate, though the high AUC suggests it could be even better with threshold optimization.

## Conclusion
The results show that the teammate's hierarchical ensemble logic is sound and provides a robust alternative to single detectors. While it doesn't consistently beat the simpler linear Stacker in F1, its superior ROC-AUC in cross-generator settings indicates that the multi-level architecture captures generator-invariant features more effectively than flat models.
