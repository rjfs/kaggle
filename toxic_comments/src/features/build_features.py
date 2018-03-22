import click
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

swear_words = [
    'ass',
    'asshole',
    'bitch',
    'cunt',
    'fuck',
    'fucker',
    'fucking',
    'fucked',
    'motherfucker',
    'shit',
    'dick',
    'piss',
    'crap',
    'cock',
    'pussy',
    'fag',
    'faggot',
    'bastard',
    'slut',
    'suck',
    'sucker',
    'anal'
]

hate_words = [
    'muslim',
    'nigga',
    'gay',
    'virgin',
    'arab',
    'arabian',
    'catholic',
    'jesus',
    'christians',
    'homosexual',
    'homo',
    'hell',
    'mother',
    'palestinians',
    'hitler',
    'nazi',
    'russia',
    'russian'
]


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    print('Reading raw data...')
    raw_train = pd.read_csv(input_dir + 'train.csv', index_col='id')
    raw_test = pd.read_csv(input_dir + 'test.csv', index_col='id')
    train_features = get_features(raw_train)
    test_features = get_features(raw_test)
    print('Writing output...')
    train_features.to_csv(output_dir + 'train-features.csv')
    test_features.to_csv(output_dir + 'test-features.csv')


def get_features(data):
    features = pd.DataFrame(index=data.index)
    functs = {
        'n_upper': count_upper,
        'n_lower': count_lower,
        'n_stars': count_stars,
        'n_smilies': count_smilies,
        'n_punctuation': count_punctuation,
        'n_points': count_point,
        'n_comma': count_comma,
        'n_symbols': count_symbols,
        'n_exclamation': count_exclamation,
        'n_interrogation': count_interrogation,
        'has_ip': has_ip,
        'sentiment': vader_sentiment
    }
    for c, f in functs.items():
        print('Computing \'%s\'' % c)
        features[c] = data['comment_text'].apply(f)

    token_functs = {
        'n_words': count_words,
        'n_unique': count_unique,
        'n_swears': count_swears,
        'n_hate': count_hates
    }
    words_lst = [nltk.word_tokenize(t) for t in data['comment_text'].values]
    for c, f in token_functs.items():
        print('Computing \'%s\'' % c)
        features[c] = pd.Series(words_lst, index=data.index).apply(f)

    return features


def count_punctuation(text):
    return sum(text.count(p) for p in '.,;:')


def count_symbols(text):
    return sum(text.count(s) for s in '*&$%')


def count_smilies(text):
    return sum(text.count(w) for w in (':-)', ':)', ';-)', ';)', ':(', ':/'))


def count_upper(text):
    return sum([x.isupper() for x in text])


def count_lower(text):
    return sum([x.islower() for x in text])


def count_stars(text):
    return sum([x == '*' for x in text])


def count_exclamation(text):
    return sum([x == '!' for x in text])


def count_interrogation(text):
    return sum([x == '?' for x in text])


def count_point(text):
    return sum([x == '.' for x in text])


def count_comma(text):
    return sum([x == ',' for x in text])


def capital_letters_pct(text):
    """ Computes percentage of capital letters in given text, not counting with spaces"""
    n_caps = sum([x.isupper() for x in text])
    n_lower = sum([x.islower() for x in text])
    if n_caps == 0 and n_lower == 0:
        return 0.0
    return float(n_caps) / (n_lower + n_caps)


def count_words(words):
    return len(words)


def count_unique(words):
    return len(set(words))


def count_swears(words):
    return sum([w.lower() in swear_words for w in words])


def count_hates(words):
    return sum([w.lower() in hate_words for w in words])


def unique_words_pct(text):
    """ Computes percentage of unique words, in given text """
    words = nltk.word_tokenize(text)
    n_unique = len(set(words))
    if len(words) == 0:
        return 0.0
    else:
        return float(n_unique) / len(words)


def has_ip(text):
    ip = re.findall(r'[0-9]+(?:\.[0-9]+){3}', text)
    return len(ip) > 0


def vader_sentiment(text):
    return sid.polarity_scores(text)['compound']


if __name__ == '__main__':
    main()
