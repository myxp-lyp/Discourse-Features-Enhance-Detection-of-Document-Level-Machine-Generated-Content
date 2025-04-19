from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt


def calculate_sentence_lengths(sentences):
    # Tokenize each sentence
    #tokenized_sentences = [sent_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [sentences]
    # Flatten the list of lists into a single list of sentences
    sentences_flat = [sentence for sublist in tokenized_sentences for sentence in sublist]

    # Calculate lengths of each sentence
    sentence_lengths = [len(sentence.split()) for sentence in sentences_flat]

    # Calculate average, maximum, and minimum sentence lengths
    print(sum(sentence_lengths), len(sentence_lengths))
    average_length = sum(sentence_lengths) / len(sentence_lengths)
    
    max_length = max(sentence_lengths)
    min_length = min(sentence_lengths)

    return average_length, max_length, min_length

from nltk import pos_tag
nltk.download('universal_tagset')
def calculate_pos_ratios(sentences):
    # Tokenize each sentence and perform part-of-speech tagging
    tagged_sentences = [pos_tag(word_tokenize(sentence),tagset='universal') for sentence in sentences]

    # Flatten the list of lists into a single list of (word, pos) tuples
    words_and_pos_flat = [(word.lower(), pos) for sublist in tagged_sentences for word, pos in sublist]

    # Filter out punctuation and count the total number of types
    words_filtered = [word for word, pos in words_and_pos_flat if pos.isalpha()]

    # Calculate the total number of tokens and types
    total_tokens = len(words_filtered)

    # Calculate counts for each part of speech
    pos_counts = nltk.FreqDist(pos for _, pos in words_and_pos_flat if pos.isalpha())

    # Calculate ratios for each part of speech
    pos_ratios = {pos: count / total_tokens for pos, count in pos_counts.items()}

    return pos_ratios

def calculate_lexical_diversity(text):
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    total_types = len(set(tokens))
    lexical_diversity = total_types / total_tokens if total_tokens > 0 else 0
    return lexical_diversity

def average_lexical_diversity(texts):
    # Calculate lexical diversity for each text in the list
    lexical_diversities = [calculate_lexical_diversity(text) for text in texts]

    # Calculate the average lexical diversity
    average_diversity = sum(lexical_diversities) / len(lexical_diversities) if len(lexical_diversities) > 0 else 0

    return average_diversity
# Sample text data (multiple documents)
file_path = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/data_analysis/HPC'  # 替换为实际的文件路径
#file_path = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/test.HPC'
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
with open(file_path, 'r', encoding='utf-8') as file:
    documents = file.read().splitlines()
    
# Tokenization and Stopwords removal
stop_words_nltk = set(stopwords.words('english'))
additional_stopwords = set([
    'would', 'could', 'should', 'might',
    'a', 'an', 'the',
    'one', 'it', 'they',
    'and', 'but', 'or', 'nor',
    'very', 'really', 'quite',
    'some', 'any', 'many', 'few',
    'now', 'today', 'yesterday', 'tomorrow','however','even'
])
stop_words = stop_words_nltk.union(additional_stopwords)
all_tokens = []



for doc in documents:
    sentences = sent_tokenize(doc)
    doc_tokens = [word_tokenize(sentence) for sentence in sentences]
    doc_tokens = [word.lower() for sublist in doc_tokens for word in sublist if word.isalpha() and word.lower() not in stop_words]
    all_tokens.extend(doc_tokens)

# Frequency distribution of words
fdist = FreqDist(all_tokens)
words, frequencies = zip(*fdist.most_common(20))
plt.bar(words, frequencies, color='skyblue')
plt.xlabel('HPC_Words')
plt.ylabel('Frequencies')
plt.title('Top 20 Most Common Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure as an image file (e.g., PNG)
plt.savefig('HPC_word_frequencies.png')

#average_length, max_length, min_length = calculate_sentence_lengths(documents)

#print(average_length, max_length, min_length)


ratios_by_pos = calculate_pos_ratios(documents)
for pos, ratio in ratios_by_pos.items():
    print(f"{pos.capitalize()} Ratio: {ratio:.4f}")
    
    
#average_ld = average_lexical_diversity(documents)

# Print the result
#print(f"Average Lexical Diversity: {average_ld:.4f}")
