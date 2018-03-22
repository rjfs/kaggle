"""
https://stackoverflow.com/questions/29099621/how-to-find-out-wether-a-word-exists-in-english-using-nltk
"""
import nltk
import pandas as pd
from itertools import chain
from collections import defaultdict

VERIFIED_WORDS_FILE = 'verified.txt'
MAPPING_WORDS_FILE = 'mapping.csv'


def main():
    f = '../../data/raw/train.csv'
    create_word_mappings(f)


class UnusualWords:

    def __init__(self):
        self.english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        self.verified = set(get_verified_words())

    def unusual_words(self, text):
        text_vocab = set(w.lower() for w in text.split())
        unusual = text_vocab - self.english_vocab - self.verified
        return sorted(unusual)


def get_verified_words():
    out = []
    with open(VERIFIED_WORDS_FILE, 'r') as f:
        for line in f:
            for word in line.split():
                out.append(word)

    return out


def create_word_mappings(f):
    df = pd.read_csv(f, index_col='id')
    uw = UnusualWords()
    uw_list = [uw.unusual_words(str(i)) for i in df['comment_text'].values]
    word_freq = frequency(*uw_list)
    with open(VERIFIED_WORDS_FILE, 'a') as f_ver:
        with open(MAPPING_WORDS_FILE, 'a') as f_map:
            for w, n in word_freq:
                user_w = input('[%d] \'%s\': ' % (n, w))
                if user_w != '':
                    # Write to mapping
                    f_map.write('%s,%s\n' % (w, user_w))
                # Write to verified
                f_ver.write('%s\n' % w)


def frequency(*lists):
    counter = defaultdict(int)
    for x in chain(*lists):
        counter[x] += 1

    return sorted(
        counter.items(),
        key=lambda kv: (kv[1], kv[0]),
        reverse=True
    )


if __name__ == '__main__':
    main()
