# Unified Benchmark Dataset Report

## Dataset Summary
- **Controlled Dataset**: 600 samples (Easy / controlled single-generator setting)
- **HC3 Dataset**: 4000 samples (Medium / semi-realistic QA setting)
- **M4 Dataset**: 14370 samples (Hard / multi-domain multi-generator setting)
- **Total Combined**: 18970 samples

## Distribution Analysis
### Label Balance
- **Controlled**: 300 Human, 300 AI
- **HC3**: 2000 Human, 2000 AI
- **M4**: 7185 Human, 7185 AI

### Domain Diversity
- **Controlled**: essay, wikipedia, news
- **HC3**: QA
- **M4**: arxiv, peerread, wikipedia, reddit_eli5, wikihow

### Generator Diversity
- **Controlled**: human, llm
- **HC3**: human, chatgpt
- **M4**: bloomz, human, davinci003, dolly, chatgpt, cohere

## Expected Experimental Outcomes
- The Controlled dataset provides a clean signal, making it the easiest benchmark where signal-based models should perform best.
- The HC3 dataset introduces a moderate distribution shift with longer and more conversational QA texts. Signal models may see a performance drop.
- The M4 dataset presents the hardest challenge due to its extreme multi-domain and multi-generator nature, causing significant feature overlap between Human and AI.
- Transformer-based models are expected to be more robust across the cross-dataset generalizations than lexical or pure signal-based baseline models.

