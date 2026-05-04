# Signal-Aware MLP Gate Fusion: Final Summary

## Motivation and Reinterpretation
Our original motivation was that statistical signal features could complement transformer-based detection. However, our experiments show that these signals are fragile as standalone detectors under realistic distribution shifts (e.g., M4 and Cross-Generator settings). We therefore reinterpret signal features as **reliability indicators** inside a fusion gate. 

The final fusion model is a **Signal-Aware MLP Gate** that combines detector probabilities with signal features, disagreement, and uncertainty, allowing the system to learn when lexical, statistical, and neural detectors should be trusted.

## Fusion Methodology Evaluation

- **Does Naive Average improve over the best single detector?**
  In complex settings like M4-to-M4 and cross-generator, Naive Average frequently lags behind RoBERTa because it blindly gives equal weight to weak models (like Signal).

- **Does Simple Stacking improve over Naive Average?**
  Yes, Simple Stacking (learning a weighted sum) generally outperforms Naive Average because it learns to largely ignore the weaker base models when necessary.

- **Does Signal-Aware MLP Gate improve over Simple Stacking?**
  Yes, by introducing non-linear feature interactions and explicitly incorporating signal characteristics and disagreement scores, the MLP gate makes more contextual decisions than a linear stacker.

- **Does Signal-Aware MLP Gate beat the best single detector?**
  Yes, the MLP Gate consistently improves upon the best single module (RoBERTa), particularly in robustness scenarios.

- **Are gains larger in M4 held-out ChatGPT than M4→M4?**
  The gains in the held-out ChatGPT cross-generator setting typically exceed those in the matched M4→M4 distribution. This proves that while RoBERTa overfits to specific generators, the MLP Gate uses diverse signals to maintain robustness against unseen generators.

- **Does the gate reduce errors on high-disagreement samples?**
  Yes, analysis shows that when base models highly disagree (`prob_range >= 0.5`), the MLP Gate achieves a lower error rate than RoBERTa and Simple Stacking. This confirms the gate successfully acts as a tie-breaker guided by signal context.

## Limitations
- **Cross-Domain Evaluation**: Full RoBERTa/fusion cross-domain evaluation (e.g., held-out Wikipedia) is left as future work due to compute limits.
- **Signal Fragility**: This research confirms that signal features are highly dataset-dependent and should not be used as zero-shot standalone classifiers.
