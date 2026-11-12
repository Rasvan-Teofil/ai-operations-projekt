"""Discarded full-window inference capacity check on repeated Train text only."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from src.next_stage_experiment import ART, OUT, TRAIN, gpu, clean, verified
from src.next_stage_collect import write_json
from src.next_stage_prepare import frozen_write
from src.serving import load_artifact


def probe():
    verified();gpu();table=pd.read_csv(TRAIN)
    predictor=load_artifact(ART/'cv/fold0/qwen-lora')
    encoded=[predictor.tokenizer.encode(text,add_special_tokens=False,verbose=False) for text in table.text]
    index=max(range(len(encoded)),key=lambda i:len(encoded[i]));source=encoded[index]
    ids=(source*(4096//len(source)+1))[:4096];text=predictor.tokenizer.decode(ids)
    # Encode/decode boundary merges can change token count. Shorten only this
    # artificial capacity input until it is inside the already frozen limit.
    while len(predictor.tokenizer.encode(text,add_special_tokens=False,verbose=False))>4096:
        ids=ids[:-1];text=predictor.tokenizer.decode(ids)
    message=predictor.tokenizer.apply_chat_template(predictor.messages(text),tokenize=False,add_generation_prompt=True)
    sequence=len(predictor.tokenizer.encode(message,add_special_tokens=False,verbose=False))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();scores=predictor.predict_proba([text])[0];torch.cuda.synchronize()
    if not np.isfinite(scores).all() or not np.isclose(scores.sum(),1.,atol=1e-5):raise RuntimeError('Invalid engineering inference distribution')
    report={'engineering_only':True,'scored_evaluation':False,'source':'Repeated and bounded longest Train article; not a natural new data item and never added to Train/Test',
        'source_train_id':table.iloc[index].id,'source_train_text_sha256':table.iloc[index].text_sha256,'artificial_input_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'article_tokens':len(predictor.tokenizer.encode(text,add_special_tokens=False,verbose=False)),'full_sequence_tokens':sequence,
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'profile':'CV fold0 Qwen LoRA, same full-window inference implementation and dependencies as final candidates','truncation':False,
        'limits':'Single artificial full-window inference check; training capacity was measured only on the actual longest fixed Train input.'}
    frozen_write(OUT/'inference-capacity-v1.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    del predictor;clean()


if __name__=='__main__':probe()
