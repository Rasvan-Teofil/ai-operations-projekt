from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from src.tokenization import token_chunks


def test_generic_fast_tokenizer_overflow_preserves_specials_and_all_tokens():
    backend = Tokenizer(WordLevel({'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'word':4},unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='<s> $A </s>', special_tokens=[('<s>',0),('</s>',2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>',eos_token='</s>',
                                       pad_token='<pad>',unk_token='<unk>')
    chunks = token_chunks(' '.join(['word']*300),tokenizer,128)
    assert [len(c['input_ids']) for c in chunks] == [128,128,50]
    assert sum(c['input_ids'].count(4) for c in chunks) == 300
    for chunk in chunks:
        assert chunk['input_ids'][0] == 0
        assert chunk['input_ids'][-1] == 2
        assert len(chunk['input_ids']) == len(chunk['token_type_ids']) == len(chunk['attention_mask'])
