"""Use backend postprocessing so generic Fast tokenizers retain their special tokens."""

def token_chunks(text, tokenizer, max_length):
    encoded = tokenizer(text, truncation=True, max_length=max_length,
                        return_overflowing_tokens=True, padding=False)
    return [{key: encoded[key][i] for key in tokenizer.model_input_names if key in encoded}
            for i in range(len(encoded['input_ids']))]
