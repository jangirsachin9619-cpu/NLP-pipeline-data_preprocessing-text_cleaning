import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK data is downloaded (silent if already present)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

slang_dict = {
    'r': 'are',
    'gr8': 'great',
    'u': 'you',
    'ur': 'your',
    'lol': 'laughing out loud',
    'brb': 'be right back',
    'btw': 'by the way'
}

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_emails(text):
    return re.sub(r'\S*@\S*\s?', '', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"😀-🙏"
                               u"🌀-🗿"
                               u"🚀-🛿"
                               u"🇠-🇿"
                               u"✂-➰"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_repeated_chars(text):
    # Replaces 3 or more repeated characters with 2 repetitions
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def replace_slang(text, slang_dictionary):
    for slang, replacement in slang_dictionary.items():
        # Use word boundaries to only replace whole words
        text = re.sub(r'\b' + re.escape(slang) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text

def full_preprocess_pipeline(raw_text):
    # 1. Lowercase
    text = raw_text.lower()

    # 2. Remove HTML tags
    text = remove_html_tags(text)

    # 3. Remove URLs
    text = remove_urls(text)

    # 4. Remove emails
    text = remove_emails(text)

    # 5. Remove emojis
    text = remove_emojis(text)

    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 7. Remove punctuation and special characters (keeping only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Normalize repeated characters
    text = normalize_repeated_chars(text)

    # 10. Replace slang
    text = replace_slang(text, slang_dict)

    # 11. Word tokenize
    tokens = word_tokenize(text)

    # 12. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 13. Stemming
    stemmed_tokens = [ps.stem(word) for word in tokens]

    return stemmed_tokens

