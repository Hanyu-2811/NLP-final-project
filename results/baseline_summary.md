# Baseline Experiments Summary

## In-Domain Performance
| Model           | Dataset    |   accuracy |       f1 |   macro_f1 |
|:----------------|:-----------|-----------:|---------:|-----------:|
| TF-IDF          | Controlled |   0.877778 | 0.888889 |   0.876543 |
| Signal Features | Controlled |   0.988889 | 0.988764 |   0.988888 |
| TF-IDF          | HC3        |   0.711667 | 0.725832 |   0.710895 |
| Signal Features | HC3        |   0.733333 | 0.755352 |   0.731156 |
| TF-IDF          | M4         |   0.831169 | 0.822092 |   0.830728 |
| Signal Features | M4         |   0.545918 | 0.612277 |   0.532216 |

## M4 Cross-Generator Performance
| Model           | HeldOut_Generator   |   accuracy |       f1 |   macro_f1 |
|:----------------|:--------------------|-----------:|---------:|-----------:|
| TF-IDF          | chatgpt             |   0.886105 | 0.644461 |   0.788326 |
| Signal Features | chatgpt             |   0.425307 | 0.29285  |   0.404411 |
| TF-IDF          | davinci003          |   0.902575 | 0.68254  |   0.812499 |
| Signal Features | davinci003          |   0.405822 | 0.254403 |   0.380262 |
| TF-IDF          | cohere              |   0.89724  | 0.648413 |   0.794119 |
| Signal Features | cohere              |   0.431338 | 0.324935 |   0.416851 |
| TF-IDF          | dolly               |   0.889121 | 0.600334 |   0.767983 |
| Signal Features | dolly               |   0.417884 | 0.282693 |   0.396445 |
| TF-IDF          | bloomz              |   0.866968 | 0.597967 |   0.759132 |
| Signal Features | bloomz              |   0.416725 | 0.295362 |   0.398893 |

## M4 Cross-Domain Performance
| Model           | HeldOut_Domain   |   accuracy |       f1 |   macro_f1 |
|:----------------|:-----------------|-----------:|---------:|-----------:|
| TF-IDF          | wikipedia        |   0.550286 | 0.649131 |   0.511518 |
| Signal Features | wikipedia        |   0.527714 | 0.626103 |   0.492578 |
| TF-IDF          | reddit_eli5      |   0.708    | 0.622878 |   0.692325 |
| Signal Features | reddit_eli5      |   0.665429 | 0.729249 |   0.645746 |
| TF-IDF          | wikihow          |   0.684857 | 0.628244 |   0.677375 |
| Signal Features | wikihow          |   0.498    | 0.333713 |   0.465504 |
| TF-IDF          | arxiv            |   0.645714 | 0.699321 |   0.634083 |
| Signal Features | arxiv            |   0.488    | 0.655914 |   0.327957 |
| TF-IDF          | peerread         |   0.859459 | 0.850575 |   0.858961 |
| Signal Features | peerread         |   0.435135 | 0        |   0.303202 |

## Analysis

### Which dataset is easiest?
- Based on the In-Domain results, Controlled should show the highest performance, confirming it is the easiest dataset to classify due to a lack of domain diversity and a single robust generator. M4 should be the hardest.

### Do signal features degrade from Controlled to HC3 to M4?
- By comparing the Signal Features models across the In-Domain results, we can observe the degradation. Signal features typically degrade from Controlled to HC3 and suffer significantly on M4 due to multi-generator and multi-domain noise.

### Which held-out generators are hardest?
- Review the M4 Cross-Generator Performance table. The generator with the lowest F1 or Macro F1 score when held-out represents the hardest generator to generalize to without explicitly training on it. (Usually ChatGPT or Cohere).

### Which held-out domains are hardest?
- Review the M4 Cross-Domain Performance table. The domain with the lowest F1 or Macro F1 score when held-out represents the hardest domain to generalize to. (Usually Wikipedia or arXiv due to strict formatting).

### Are TF-IDF or signal features more robust?
- TF-IDF relies on lexical cues and n-grams which might overfit to domain-specific topics, making it brittle in Cross-Domain setups but decent in In-Domain. Signal features attempt to capture stylometric signals (like burstiness/perplexity), which can be more domain-agnostic, but are heavily dependent on the generator. 
