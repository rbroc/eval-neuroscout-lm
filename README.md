### Evaluating LLMs on forward language modeling for NeuroScout transcripts
- Data is Neuroscout transcripts
	- Either as is
	- Or with added punctuation
- Evaluate different context windows
- Evaluate the perplexity/surprisal/entropy of the predictions
- Also qualitatively inspect the predictions models produce

Models to test:
- GPT2
- BERT / Roberta / DistilBERT / Albert
- XLM? Or other?

Things to consider
- Other dataset (e.g., Narratives, ParanoiaStory)
- Transcripts look bad - we may need to evaluate that
- Dialogue models could be more accurate in these cases. Good reasons to look into speaker segmentation.
- Also that we should have some sentiment metric;
- Get perplexity of next turns for dialogue, if it is somehow encoded in srts? (do we have srt for all of them?)