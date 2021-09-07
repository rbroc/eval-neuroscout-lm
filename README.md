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
- Other dataset (e.g., narratives?)
- transcripts may need to be re-processed or aligned with the actual text
- dialogue models could be more accurate in these cases. Good reasons to look into speaker segmentation.
- also that we should have some sentiment metric thing (different types!)
- evaluate transcripts (e.g., by presence of each word, somehow?)
- get perplexity of next turns for dialogue, if it is somehow encoded in srts? (do we have srt for all of them?)