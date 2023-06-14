from django.shortcuts import render
from django.http import JsonResponse
from .algorihms.transformer import *
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
sw_nltk = stopwords.words('english')

nltk.download('punkt')
def t5_summary(request):
   # text = "According to consensus in modern genetics, anatomically modern humans first arrived on the Indian subcontinent from Africa between 73,000 and 55,000 years ago.[1] However, the earliest known human remains in South Asia date to 30,000 years ago. Sedentariness, which involves the transition from foraging to farming and pastoralism, began in South Asia around 7000 BCE. At the site of Mehrgarh, its presence can be documented, with evidence of domestication of wheat and barley, rapidly followed by that of goats, sheep, and cattle.[2] By 4500 BCE, such settled life had increasingly spread, [2] and began to gradually evolve into the Indus Valley civilisation, which was contemporaneous with Ancient Egypt and Mesopotamia. This civilisation flourished between 2500 BCE and 1900 BCE in present-day Pakistan and north-western India, and was noted for its urban planning, baked brick houses, elaborate drainage, and water supply.[3] Early on in the second millennium BCE, persistent drought caused the population of the Indus Valley to scatter from large urban centres to villages. Around the same time, Indo-Aryan tribes moved into the Punjab from Central Asia in several waves of migration. The Vedic Period (1500–500 BCE) was marked by the composition of their large collections of hymns called Vedas. Their varna system, which evolved into the caste system, consisted of a hierarchy of priests, warriors, free peasants, and servants. The pastoral and nomadic Indo-Aryans spread from the Punjab into the Gangetic plain, large swaths of which they deforested for agriculture. The composition of Vedic texts ended around 600 BCE, when a new, interregional culture arose. Then, small chieftaincies (janapadas) were consolidated into larger states (mahajanapadas)."
   url = request.GET.get('url')
   text = text_extractor(url)
   text = clean_rawtext(text)
   summary = transformer_t5(text)
   return JsonResponse({'summary':summary})

def clean_rawtext(text):
    stoplist = set(stopwords.words("english"))
    text=re.sub(r'\w*\d\w*', '', text).strip()
    text = [word for word in text.split() if word.lower() not in sw_nltk]
    text = " ".join(text)
    text=[word for word in text.split() if word not in stoplist]
    text = " ".join(text)
    text=re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()
    text=text.replace('–','')
    return text

def text_extractor(base_url):
    r = requests.get(base_url)
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    paras = []
    for paragraph in soup.find_all('p'):
        paras.append(str(paragraph.text))

    heads = []
    for head in soup.find_all('span', attrs={'mw-headline'}):
        heads.append(str(head.text))

    text = [val for pair in zip(paras, heads) for val in pair]
    text = ' '.join(text)

    return text