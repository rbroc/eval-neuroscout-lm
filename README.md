### Evaluating LLMs on forward language modeling for NeuroScout transcripts
- Evaluate:
	- Transcripts vs. aligned (vs. scrambled?)
	- Different context windows
	- Different models (GPT, BERT, BigBird, etc).
- Metrics
	- Perplexity/surprisal/entropy of the predictions -  ** can it be done? **
    - Correct predictions on different word classes
	- Qualitatively inspect model predictions
- Datasets
	- Narratives (large-scale)
    - RBP (TO DO)

To dos (consider asking Alex):
- Subset by words in common
- Evaluate is_top_true across different categories (how to set that up?)
- Evaluate is_top_true across frequencies
- Evaluate on RBP / scrambled text too
- Can the models be compared on some intrinsic MLM/CLM metric?
- Does prediction improve when turn-separator are inserted correctly (bigbird?)

Potential uses
- Ingest losses + entropies: which one best models prediction errors and uncertainty?
    - R-squared approach
- Potentially compare transcript and force-aligned predictions on brain
- Fit with different window lengths, which could capture prediction at different hierarchical levels
- Read up resources - Note that the Narratives dataset is made for benchmarking!

Notes:
- Missing values in transcripts (NAs in Narratives, missing in ingested transcripts?)
- We may need to evaluate transcripts
- Unrelated
    - Sentiment metrics?
    - Next turn perplexity