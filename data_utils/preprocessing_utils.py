import re
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
import string
from tqdm import tqdm
import random
import spacy
import stanza
# stanza.download("uk")

class Preprocessing:

    @classmethod
    def drop_hashtags(cls, text):
        text =re.sub(r'(?<=#)\w+', "", text)
        return text

    @classmethod
    def drop_mentions(cls, text):
        text = re.sub(r'(?<=@)\w+', "",text)
        return text

    @classmethod
    def drop_url(cls, text):
        text = re.sub('https?://\S+|www\.\S+', '', text)
        return text

    @classmethod
    def drop_email(cls, text):
        text = re.sub(r'[\w\.-]+@[\w\.-]+' ,'' ,text)
        return text

    @classmethod
    def preprocess(cls,text):
        text = text.lower().strip()
        text = cls.drop_hashtags(text)
        text = cls.drop_mentions(text)
        text = cls.drop_email(text)
        text = cls.drop_url(text)
        text = text.replace("#", " ").replace("@", " ")

        return text


class SpanishTokenizerLemmatizer:
    def __init__(self):
        self.model = spacy.load("es")
    def __call__(self, text):
        return self.get_lemmas(text)
    def get_lemmas(self, text):
        doc = self.model(text)
        lemmas = [x.lemma_ for x in doc]
        return lemmas


class UkrainianTokenizerLemmatizer:
    def __init__(self, use_gpu=True):
        self.model = stanza.Pipeline("uk", use_gpu=use_gpu, processors="tokenize,lemma")
    def __call__(self, text):
        return self.get_lemmas(text)
    def get_lemmas(self, text):
        doc = self.model(text)
        lemmatized_output = []
        for s in doc.sentences:
            for t in s.words:
                lemmatized_output.append(t.lemma)

        return lemmatized_output

class PreprocessOHE:

    stop_words = get_stop_words("en") + get_stop_words("uk")
    punct = [x for x in string.punctuation] + ['’', "..."]

    @classmethod
    def filter_tokens_voc(cls, tokens, voc_dict):
        filtered_tokens = [x for x in tokens if x in voc_dict]
        if not filtered_tokens:
            return None
        return filtered_tokens

    @classmethod
    def get_encoding_dict(cls, tokenized_sents, voc_config, drop_stopword=True, drop_punct=True, add_tokens=()):
        stop_words = []
        if drop_stopword:
            stop_words += cls.stop_words
        if drop_punct:
            stop_words += cls.punct

        count = CountVectorizer(
            analyzer="word",
            tokenizer=lambda x: x,
            max_df=voc_config["max_df"],
            min_df=voc_config["min_df"],
            lowercase=False,
            stop_words=stop_words
        )

        count.fit(tokenized_sents)
        vocabulary = count.vocabulary_
        vocabulary = list(vocabulary.keys())

        print(f"Total voc length: {len(vocabulary)}")

        encoding_dict = {
            k: v for v, k in enumerate(vocabulary)
        }

        for t in add_tokens:
            encoding_dict[t] = len(encoding_dict)


        return encoding_dict



    @classmethod
    def filter_by_voc(cls, tokenized_sents, untok_sents, enc_dic, config):
        filtered_sents = []
        filtered_tok_sents = []

        for i, s in tqdm(enumerate(tokenized_sents)):
            encoded_sents = [cls.is_in_voc(w, enc_dic) for w in s]
            encoded_sents = [x for x in encoded_sents if x]
            if len(encoded_sents)/len(s) >= config["min_encoded_ratio"]:
                filtered_sents.append(untok_sents[i])
                filtered_tok_sents.append(s)
        return filtered_sents, filtered_tok_sents

    @classmethod
    def is_in_voc(cls, w, dict_):
        if w in dict_:
            return True
        else:
            return False

def apply_dropout(tokenized_sents, rate, mask_token):
    indexes_drop = random.choices(
        list(range(len(tokenized_sents))),
        k=int(rate * len(tokenized_sents))
    )

    tokenized_sents = [x if i not in indexes_drop else mask_token for i,x in enumerate(tokenized_sents)]
    return tokenized_sents