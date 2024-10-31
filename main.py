import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from collections import defaultdict

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
additional_stop_words = {'thou', 'thy', 'thee', 'shall', 'unto', 'ye', 'nephi', 'alma', 'god',
                         'lord'}  # can add more stop words
combined_stop_words = stop_words.union(additional_stop_words)
lemmatizer = WordNetLemmatizer()


class ReligiousTextAnalyzer:
    def __init__(self, csv_path):
        # initialize the analyzer with the CSV file
        self.df = pd.read_csv(csv_path)
        self.documents = self.df['Text'].tolist()
        self.names = self.df['Name_of_Text'].tolist()
        self.topics_by_document = {}
        self.topic_similarities = None

    def preprocess(self, text):
        text = re.sub(r'\W', ' ', text)
        words = nltk.word_tokenize(text.lower())
        filtered_words = [lemmatizer.lemmatize(word) for word in words
                          if word.isalpha() and word not in combined_stop_words]
        return filtered_words

    def generate_ngrams(self, texts, n=3):

        bigram = Phrases(texts, min_count=3, threshold=10)
        trigram = Phrases(bigram[texts], threshold=10)
        texts = [trigram[bigram[text]] for text in texts]
        filtered_texts = [[word for word in text if word not in combined_stop_words
                           and '_' not in word] for text in texts]
        return filtered_texts

    def extract_topics(self):
        """extract topics using LDA approach"""
        for i, document in enumerate(self.documents):
            processed_doc = self.preprocess(document)
            processed_doc = self.generate_ngrams([processed_doc])[0]

            dictionary = corpora.Dictionary([processed_doc])
            corpus = [dictionary.doc2bow(processed_doc)]

            lda_model = models.LdaModel(corpus, num_topics=5,
                                        id2word=dictionary, passes=15)

            topics = []

            # extract top 10 words for each topic
            for idx, topic in lda_model.print_topics(-1): # -1 means all topics
                topic_words = [word.split('*')[1].strip('"')
                               for word in topic.split(' + ')]
                topic_words = [word for word in dict.fromkeys(topic_words)
                               if word not in combined_stop_words][:10]
                while len(topic_words) < 10:
                    topic_words.append("")
                topics.append(', '.join(topic_words))

            self.topics_by_document[self.names[i]] = topics
        return self.topics_by_document

    def calculate_topic_similarities(self):

        """calculate similarity matrix between documents based on their topics"""
        texts = list(self.topics_by_document.keys())
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))

        # create a flat list of all topics for each text
        text_topics = {text: ' '.join(topics)
                       for text, topics in self.topics_by_document.items()}

        # calculate tf-idf vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_topics.values())

        # calculate cosine similarities
        for i in range(n_texts):
            for j in range(n_texts):
                if i != j:
                    similarity = 1 - cosine(
                        tfidf_matrix[i].toarray().flatten(),
                        tfidf_matrix[j].toarray().flatten()
                    )
                    similarity_matrix[i, j] = similarity

        self.topic_similarities = pd.DataFrame(
            similarity_matrix,
            index=texts,
            columns=texts
        )
        # return the similarity matrix
        return self.topic_similarities

    def analyze_environmental_terms(self, environmental_categories):
        """
        analyze frequency of environmental terms in each text

        environmental_categories: dict of category names and related terms

        e.g., {'water': ['water', 'river', 'ocean', 'rain'],
               'agriculture': ['farm', 'crop', 'harvest', 'field']}
        """
        results = defaultdict(dict)
        # calculate frequency of each term in each category
        for doc_name, document in zip(self.names, self.documents):
            words = self.preprocess(document)
            total_words = len(words)

            # count frequency of each term in each category
            for category, terms in environmental_categories.items():
                count = sum(1 for word in words if word in terms)
                frequency = (count / total_words) * 1000  # per 1000 words
                results[doc_name][category] = frequency

        # return results as a dataFrame
        return pd.DataFrame(results).T
    def create_visualizations(self, env_data=None):
        """
        Create visualizations for each analysis component
        only used 2 for presentation
        """

        # 1. topic similarity Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.topic_similarities,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            square=True
        )
        plt.title('Topic Similarities Between Religious Texts', pad=20, size=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        # save as image
        plt.savefig('topic_similarities.png', bbox_inches='tight')

        plt.show()

        if env_data is not None:

            # 2. environmental term frequencies
            plt.figure(figsize=(15, 8))
            ax = env_data.plot(kind='bar', width=0.8)
            plt.title('Environmental Term Frequencies by Text', pad=20, size=14)
            plt.xlabel('Religious Texts', size=12)
            plt.ylabel('Frequency per 1000 words', size=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Environmental Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # save as image
            plt.savefig('environmental_term_frequencies.png', bbox_inches='tight')

            plt.tight_layout()
            plt.show()

            # 3. environmental terms correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = env_data.T.corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                vmin=-1, vmax=1
            )
            plt.title('Correlations Between Environmental Terms', pad=20, size=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

    def create_environmental_distribution(self, env_data):
        """
        create a distribution plot for environmental terms across texts
        """
        # melt the dataframe for easier plotting
        env_melted = env_data.reset_index().melt(
            id_vars=['Name_of_Text'],
            var_name='Environmental Category',
            value_name='Frequency'
        )

        # create box plots
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=env_melted,
            x='Environmental Category',
            y='Frequency',
            palette='Set3'
        )
        plt.title('Distribution of Environmental Terms Across Texts', pad=20, size=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Frequency per 1000 words')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def create_top_terms_plot(self, env_data, top_n=5):
        """
        create a plot showing top N texts for each environmental category
        """
        plt.figure(figsize=(15, 10))

        # number of categories
        n_categories = len(env_data.columns)
        n_rows = (n_categories + 1) // 2  # calculate number of rows needed

        for idx, category in enumerate(env_data.columns, 1):
            plt.subplot(n_rows, 2, idx)

            # sort values for this category
            top_texts = env_data[category].sort_values(ascending=False).head(top_n)

            # create horizontal bar plot
            bars = plt.barh(
                range(len(top_texts)),
                top_texts.values,
                color=plt.cm.Set3(idx / len(env_data.columns))
            )

            # add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}',
                    ha='left',
                    va='center',
                    fontweight='bold'
                )

            plt.title(f'Top {top_n} Texts - {category}')
            plt.yticks(range(len(top_texts)), top_texts.index, size=8)
            plt.xlabel('Frequency per 1000 words')

        plt.tight_layout()
        plt.show()


# define environmental terms to search for
environmental_categories = {
    'water': ['water', 'river', 'ocean', 'rain', 'sea', 'flood', 'stream', 'lake'],
    'agriculture': ['farm', 'crop', 'harvest', 'field', 'seed', 'plant', 'grow'],
    'climate': ['sun', 'wind', 'storm', 'heat', 'cold', 'season', 'weather'],
    'land': ['mountain', 'desert', 'forest', 'valley', 'hill', 'earth', 'soil'],
    'animals': ['cattle', 'sheep', 'bird', 'fish', 'beast', 'flock', 'herd']
}

# initialize the analyzer with the dataset
analyzer = ReligiousTextAnalyzer('Dataset_with_Text.csv')

# extract topics using LDA
topics = analyzer.extract_topics()

# calculate similarities between texts
similarities = analyzer.calculate_topic_similarities()

# analyze environmental terms
env_analysis = analyzer.analyze_environmental_terms(environmental_categories)

# create visualizations

analyzer.create_visualizations(env_analysis)
analyzer.create_environmental_distribution(env_analysis)
analyzer.create_top_terms_plot(env_analysis)

#Notes
""" see what topics were important for each religion outside of chosen topics """