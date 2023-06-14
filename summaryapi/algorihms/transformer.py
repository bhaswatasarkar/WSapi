from transformers import AutoTokenizer, AutoModelWithLMHead

import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
from transformers import AutoModelWithLMHead
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def transformers_t5_base(text):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
    inputs = tokenizer.encode("summarize: " + text,
    return_tensors='pt',
    max_length=512,
    truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    return summary

def transformers_t5_small(text):
    tokenizer1 = AutoTokenizer.from_pretrained('t5-small')
    my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    input_ids=tokenizer1.encode(text, return_tensors='pt', max_length=1024,truncation=True)
    summary_ids = my_model.generate(input_ids,max_length=500, min_length=80, length_penalty=5., num_beams=2)
    t5_summary = tokenizer1.decode(summary_ids[0])
    return t5_summary

def transformer_t5(text):
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

    model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

    tokens_input = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(tokens_input, min_length=80, max_length=1024)
    t5_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return t5_summary