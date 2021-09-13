### Evaluating LLMs on forward language modeling for NeuroScout transcripts
- Evaluate:
	- Transcripts vs. aligned
	- Different context windows
	- Different models (GPT, BERT, BigBird, etc).
- Metrics
	- Perplexity/surprisal/entropy of the predictions
	- Qualitatively inspect model predictions
- Datasets
	- Narratives (large-scale)

Notes:
- Missing values in transcripts (NAs in Narratives, missing in ingested transcripts?)
- We may need to evaluate transcripts
- Unrelated
    - Sentiment metrics?
    - Next turn perplexity
- Set up same comparison for reading brain project (scrambled vs. not scrambled)