#%% 
import pandas as pd
import nltk
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster

#%%
class ReligiousTextThemeAnalyzer:
    def __init__(self):
        # Initialize NLTK components
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Clean and preprocess the text"""
        # Convert to lowercase and tokenize
        tokens = nltk.word_tokenize(str(text).lower())
        
        # Lemmatize and remove stopwords
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)
    
    def chunk_text(self, text, chunk_size=300, overlap=50):
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if len(chunk) >= chunk_size // 2:  # Only keep chunks of substantial size
                chunks.append(' '.join(chunk))
        
        return chunks
    
    def analyze_single_text(self, text, text_name):
        """Analyze themes in a single religious text"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            chunks = self.chunk_text(processed_text)
            
            if len(chunks) < 2:
                print(f"Warning: {text_name} is too short for meaningful analysis")
                return None
            
            # Configure BERTopic with more lenient parameters
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                min_df=1,
                max_df=1.0
            )
            
            topic_model = BERTopic(
                vectorizer_model=vectorizer,
                min_topic_size=2,
                nr_topics="auto",
                verbose=True,
                calculate_probabilities=True
            )
            
            # Fit the model to this text's chunks
            topics, probs = topic_model.fit_transform(chunks)
            
            # Filter out -1 (outlier) topics and get unique topics
            valid_topics = [t for t in set(topics) if t != -1]
            
            if not valid_topics:
                print(f"Warning: No clear themes found in {text_name}")
                return None
            
            # Get theme information
            themes = []
            for topic_id in valid_topics:
                topic_words = topic_model.get_topic(topic_id)
                
                if not topic_words:
                    continue
                
                word_weights = [weight for _, weight in topic_words[:10]]
                theme_coherence = np.mean(word_weights)
                theme_prevalence = np.sum([1 for t in topics if t == topic_id]) / len(topics)
                
                themes.append({
                    'theme_id': topic_id,
                    'top_words': [word for word, _ in topic_words[:10]],
                    'word_weights': word_weights,
                    'coherence': theme_coherence,
                    'prevalence': theme_prevalence,
                    'representative_chunks': [
                        chunks[i] for i, t in enumerate(topics) if t == topic_id
                    ][:3]
                })
            
            return {
                'text_name': text_name,
                'themes': themes,
                'topic_model': topic_model,
                'n_chunks': len(chunks)
            }
            
        except Exception as e:
            print(f"Error analyzing {text_name}: {str(e)}")
            return None
    
    def analyze_corpus(self, df, text_column='Text', name_column='Name_of_Text'):
        """Analyze each religious text independently"""
        results = {}
        
        for _, row in df.iterrows():
            text_name = row[name_column]
            text_content = row[text_column]
            
            print(f"\nAnalyzing {text_name}...")
            analysis = self.analyze_single_text(text_content, text_name)
            
            if analysis is not None:
                results[text_name] = analysis
        
        return results
    
    def print_top_themes_per_document(self, analysis_results, top_n=10):
        """Print top N themes for each text"""
        for text_name, analysis in analysis_results.items():
            print(f"\n{'='*20} Top {top_n} Themes in {text_name} {'='*20}")
            print(f"Number of chunks analyzed: {analysis['n_chunks']}")
            
            if not analysis['themes']:
                print("No clear themes identified")
                continue
            
            # Sort themes by prevalence and take top N
            sorted_themes = sorted(
                analysis['themes'],
                key=lambda x: x['prevalence'],
                reverse=True
            )[:top_n]
            
            for i, theme in enumerate(sorted_themes, 1):
                print(f"\nTheme {i}:")
                print(f"Prevalence: {theme['prevalence']:.2%}")
                print(f"Coherence: {theme['coherence']:.3f}")
                print("Key concepts:", ', '.join(theme['top_words']))
                print("\nExample context:")
                for chunk in theme['representative_chunks'][:1]:
                    print(f"  {chunk[:200]}...")  # Show first 200 chars
                print('-' * 40)

class TopicSimilarityAnalyzer:
    def __init__(self, analysis_results, original_dataframe):
        """
        Initialize the analyzer with results from ReligiousTextThemeAnalyzer
        
        :param analysis_results: Dictionary of analysis results from ReligiousTextThemeAnalyzer
        :param original_dataframe: Original DataFrame with additional context
        """
        self.analysis_results = analysis_results
        self.df = original_dataframe
    
    def extract_topic_embeddings(self):
        """
        Extract word embeddings for topics across all texts
        
        :return: Dictionary with text names and their topic embeddings
        """
        topic_embeddings = {}
        
        for text_name, analysis in self.analysis_results.items():
            if not analysis or 'topic_model' not in analysis:
                continue
            
            # Get word embeddings from BERTopic model
            topic_model = analysis['topic_model']
            embeddings = []
            
            for theme in analysis['themes']:
                # Use the top words to create a topic embedding
                top_words = theme['top_words']
                try:
                    # Average word embeddings for the top words
                    topic_embedding = np.mean([
                        topic_model.embedding_model.embedding_model.encode(word) 
                        for word in top_words
                    ], axis=0)
                    embeddings.append((theme['theme_id'], topic_embedding))
                except Exception as e:
                    print(f"Could not create embedding for {text_name}: {e}")
            
            topic_embeddings[text_name] = embeddings
        
        return topic_embeddings
    
    def compute_topic_similarities(self, similarity_threshold=0.7):
        """
        Compute and group similar topics across texts
        
        :param similarity_threshold: Cosine similarity threshold for grouping
        :return: List of topic groups
        """
        topic_embeddings = self.extract_topic_embeddings()
        
        # Flatten all topic embeddings
        all_embeddings = []
        embedding_metadata = []
        
        for text_name, text_embeddings in topic_embeddings.items():
            for theme_id, embedding in text_embeddings:
                all_embeddings.append(embedding)
                embedding_metadata.append({
                    'text_name': text_name,
                    'theme_id': theme_id
                })
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(all_embeddings)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(similarity_matrix, method='ward')
        
        # Cluster topics
        clusters = fcluster(
            linkage_matrix, 
            t=similarity_threshold, 
            criterion='distance'
        )
        
        # Group topics by cluster
        topic_groups = {}
        for i, cluster in enumerate(clusters):
            if cluster not in topic_groups:
                topic_groups[cluster] = []
            
            metadata = embedding_metadata[i]
            topic_details = self._get_topic_details(
                metadata['text_name'], 
                metadata['theme_id']
            )
            
            topic_groups[cluster].append({
                'text_name': metadata['text_name'],
                'theme_id': metadata['theme_id'],
                'top_words': topic_details['top_words']
            })
        
        return topic_groups
    
    def _get_topic_details(self, text_name, theme_id):
        """
        Retrieve topic details for a specific text and theme
        
        :param text_name: Name of the text
        :param theme_id: ID of the theme
        :return: Dictionary with topic details
        """
        analysis = self.analysis_results[text_name]
        for theme in analysis['themes']:
            if theme['theme_id'] == theme_id:
                return theme
        return None

    def _get_shared_context(self, topics):
        """
        Find shared contextual information for topics with flexible matching
        
        :param topics: List of topics from different texts
        :return: Dictionary of shared context
        """
        # Get rows for these texts
        text_rows = self.df[self.df['Name_of_Text'].isin([t['text_name'] for t in topics])]
        
        # Find columns with shared values
        shared_context = {}
        
        # Define special handling for specific column types
        column_matching_rules = {
            # Exact match columns
            'exact_match': [
                'Country_of_Origin', 
                'Region_of_Country', 
                'Language',
                'Type_of_Religion',
                'Practition_Status',
                'Open_Closed_Status',
                'Arability',
                'Earthquake',
                'Volcanics',
                'Coastal_Flooding',
                'Water_Scarcity',
                'Extreme_Heat',
                'River_Flooding',
                'Tsunami',
                'Landslide',
                'Cyclone',
                'Wildfire'
            ],
            
            # Numeric columns with tolerance
            'numeric_tolerance': {
                'Max_Temp': 5,   # Within 5 degrees
                'Min_Temp': 5,   # Within 5 degrees
                'Elevation': 100, # Within 100 meters
                'Date_of_Text_Origin': 100, # Within 100 years
                'Rainfall': 5, # Within 5 inches of rain
                'Wind': 1, # Within 1 mph of wind speed
                'Latitude' : 2, #Within 2 degrees of latitude
                'Longitude' : 2 #Within 2 degrees longitude
            },
            
            # Columns with potential multiple values (like biomes)
            'multi_value': ['Biome', 'Climate']
        }
        
        # Check exact match columns
        for column in column_matching_rules['exact_match']:
            if column in text_rows.columns:
                unique_values = text_rows[column].unique()
                if len(unique_values) == 1:
                    shared_context[column] = unique_values[0]
        
        # Check numeric columns with tolerance
        for column, tolerance in column_matching_rules['numeric_tolerance'].items():
            if column in text_rows.columns:
                values = text_rows[column]
                if len(values) > 1:
                    # Check if all values are within the tolerance
                    min_val = values.min()
                    max_val = values.max()
                    if max_val - min_val <= tolerance:
                        shared_context[column] = f"{min_val:.1f} ± {tolerance}"
        
        # Check multi-value columns
        for column in column_matching_rules['multi_value']:
            if column in text_rows.columns:
                # Split multi-value entries and find common values
                all_values = text_rows[column].str.split(',').explode().str.strip()
                
                # Find common values across all entries
                common_values = set.intersection(
                    *[set(val.split(',')) for val in text_rows[column]]
                )
                
                if common_values:
                    shared_context[f"Shared {column}"] = list(common_values)
        
        return shared_context
    

    
    def print_topic_groups(self, topic_groups):
        """
        Print cross-text topic groups with shared context
        
        :param topic_groups: Dictionary of topic groups
        """
        for group_id, topics in topic_groups.items():
            # Check if topics are from different texts
            unique_texts = set(topic['text_name'] for topic in topics)
            
            # Only print groups with topics from multiple texts
            if len(unique_texts) > 1:
                print(f"\nGroup {group_id}:")
                
                # Find and print shared context
                shared_context = self._get_shared_context(topics)
                if shared_context:
                    print("Shared Context:")
                    for key, value in shared_context.items():
                        print(f"  {key}: {value}")
                
                # Print topic details
                for topic in topics:
                    print(f"{topic['text_name']} - Theme {topic['theme_id']}: " +
                        f"Top Words: {', '.join(topic['top_words'])}")
                print("-" * 50)

#%%
# Initialize analyzer
analyzer = ReligiousTextThemeAnalyzer()

# Load data
df = pd.read_csv('Dataset_with_Text.csv', encoding='utf-8')

# Analyze texts
results = analyzer.analyze_corpus(df)

#%%
# Perform topic similarity analysis
similarity_analyzer = TopicSimilarityAnalyzer(results, df)
topic_groups = similarity_analyzer.compute_topic_similarities(similarity_threshold=0.9)
similarity_analyzer.print_topic_groups(topic_groups)

# %%
print(df)
# %%
