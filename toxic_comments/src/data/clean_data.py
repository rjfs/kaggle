import pandas as pd
import re
import nltk

lem = nltk.stem.wordnet.WordNetLemmatizer()
tokenizer = nltk.tokenize.TweetTokenizer()
eng_stopwords = set(nltk.corpus.stopwords.words("english"))


class DataCleaner:

    def __init__(self, mapping_file):
        self.words_mapping = get_words_mapping(mapping_file)

    def clean(self, data):
        data['comment_text'] = data['comment_text'].apply(self.clean_text)
        return data

    def clean_text(self, text, lower=False):
        # to lower
        if lower:
            text = text.lower()
        # remove \n
        text = re.sub("\\n", " ", text)
        # remove leaky elements like ip,user
        text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", text)
        # removing usernames
        text = re.sub("\[\[.*\]", "", text)
        # Split the sentences into words
        words = tokenizer.tokenize(text)
        # Replace some words
        words = [
            self.words_mapping[word] if word in self.words_mapping else word
            for word in words
        ]
        words = [lem.lemmatize(word, "v") for word in words]
        words = [w for w in words if w not in eng_stopwords]

        text = " ".join(words)

        return text


def get_words_mapping(mapping_file):
    """
    Gets words mapping dictionary from given CSV file
    :param mapping_file: str
        Path to CSV words mapping file
    :return: dict
        Mapping dictionary as {raw_word: mapped_word}
    """
    df = pd.read_csv(mapping_file)
    return {k: v for k, v in df.values}
