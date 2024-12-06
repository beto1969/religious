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
    def __init__(self, custom_stopwords=None):
        # Initialize NLTK components
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Get default English stopwords
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Clean and preprocess the text"""
        # Convert to lowercase and tokenize
        tokens = nltk.word_tokenize(str(text).lower())
        
        # Remove stopwords and non-alphabetic tokens
        cleaned_tokens = [
            token
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
                calculate_probabilities=True,
                top_n_words=20
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
                
                word_weights = [weight for _, weight in topic_words[:20]]
                theme_coherence = np.mean(word_weights)
                theme_prevalence = np.sum([1 for t in topics if t == topic_id]) / len(topics)
                
                themes.append({
                    'theme_id': topic_id,
                    'top_words': [word for word, _ in topic_words[:20]],
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
    
    def print_top_themes_per_document(self, analysis_results, top_n=20):
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
    
    def compute_topic_similarities(self, similarity_threshold=0.85):
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
# Example of adding custom stopwords
custom_stops = ['arjuna', 'krishna', 'mazda', 'ahura mazda', 'brahma', 'dhammas', 'moroni',
                'jacob', 'enos', 'christ', 'unas', 'osiris', 'mosiah', 'prabhaatee mehl', 'ani',
                'text chapter', 'jared', 'brother jared', 'pharoah', 'moses', 'mallans', 'kusinr', 
                'coriantumr', 'shiz', 'pas', 'came pas', 'lib', 'qasiagssaq', 'ven', 'nanda',
                'ven nanda', 'tathgata', 'dadda', 'nanna', 'raag fifth', 'siree raag', 'dayv',
                'naam dayv', 'naam', 'arit', 'dhamma', 'shabad', 'maajh', 'vajjians', 'nanda', 
                'magadha', 'vassakra', 'jvaka', 'saarang fourth', 'shalok', 'pauree', 'pukkusa', 
                'pv', 'avesta', 'zend', 'sraosha', 'chukchee', 'eskimo', 'iii', 'fargard', 'fargards',
                'venddd', 'farg', 'abraham', 'lut', 'isaac', 'noah', 'armf', 'orsha', 'hermes', 'apollo', 
                'odin', 'njrdr', 'skadi', 'xibalba', 'hunahp', 'xbalanqu', 'yawahu', 'tohil',
                'tohil avilix', 'mahucutah', 'hacavitz', 'avilix', 'aviliz hacavitz', 'dava',
                'thrita', 'zipacn', 'cabracn', 'hunahp xbalanqu', 'dinewan', 'goombelgubbon', 'oolah',
                'dinewans', 'ha', 'wayambeh', 'galah', 'kgssagssuk', 'little kgssagssuk', 'little umerdlugtoq',
                'ra', 'salih', 'arthur', 'gangleri', 'thor', 'skrmir', 'hrungnir', 'hymir',
                'allah', 'son mary', 'mary', 'jesus', 'jesus son', 'allah lord', 'zeus', 
                'maia', 'son maia', 'son zeus', 'mus', 'dy', 'kabeer', 'inkalimeva', 'fareed',
                'mehl fareed', 'lord fareed', 'gir', 'metaneira', 'demeter', 'mugals', 'gauree',
                'raag mehl', 'sahu', 'pepi', 'ka', 'bilaaval', 'dhanaasaree fifth', 'dhanaasaree',
                'second mehl', 'lord dakhanay', 'pauree', 'hukam', 'todee fifth', 'todee',
                'vii', 'viii', 'asura', 'varuna', 'tuyallay', 'ptah', 'horus', 'maat', 'set', 'shu',
                'nut', 'seb', 'zarathustra', 'unkulunkulu', 'amadhlozi', 'amatongo', 'umancele',
                'usopetu', 'upeteni', 'itongo', 'ufaku', 'uncapayi', 'udumisa', 'rangitu',
                'kauilani', 'maui', 'pikoi', 'mainele', 'tabu', 'umyeka', 'arawak', 'ukaleq', 
                'atdlarneq', 'lumawig', 'konehu', 'll', 'll ll', 'avvang', 'neruvkq', 'sigurdr',
                'mamala', 'ouha', 'dkw', 'deereeree', 'wyah', 'bibbee', 'kamapuaa', 'pele',
                'ollantay', 'tupac yupanqui', 'tupac', 'yupanqui', 'uillac uma', 'osiris ani',
                'konehu', 'qujvrssuk', 'tupilak', 'tugto', 'pia', 'carib', 'makusis', 'arawak',
                'tutanekai', 'te', 'kawelo', 'kauai', 'namaka', 'aiyobanni', 'hatupatu', 'siwara', 
                'mawri', 'okoyumo', 'papik', 'serikoai', 'konehu', 'koneso', 'deegeenboyah', 'mullyan',
                'mullyangah', 'ah ah', 'te ponga', 'ponga', 'tawhaki', 'karihi', 'tatua', 'weedah',
                'weeoombeens', 'piggiebillah', 'bougoodoogahdah', 'bahloo', 'turi', 'arawa', 
                'manoa', 'oahu', 'nuuanu', 'kou', 'whakatau', 'seneca', 'lawrence', 'asalq', 'makte',
                'sedna', 'guillemot', 'takarangi', 'hakawau', 'puarata', 'maketu', 'komatari',
                'kororomanna', 'hebu', 'koneso', 'nafudi', 'haburi', 'hlakanyana', 'sikulume',
                'inabulele', 'mangangezulu', 'ptussorssuaq', 'altaq', 'qalagnguas', 'tallarssuaq',
                'kupe', 'turi', 'hoturapa', 'rehua', 'rupe', 'manaia', 'ihenga', 'wakea', 'hervey',
                'kamapuaa', 'kokoa', 'awa', 'manoa', 'kapuni', 'dhanna', 'gobind', 'lludd', 'isis',
                'ahura', 'nephi', 'nephites', 'giddianhi', 'gidgiddoni', 'lachoneus', 'gunnarr', 'brynhildr', 
                'gunnar hgni', 'hgni', 'gudrn', 'atli', 'gjki', 'limhi', 'ammon', 'zarahelma', 'king limhi', 
                'jvaka', 'inuence', 'pali', 'khadoor', 'pheru', 'elphin', 'taliesin', 'heinin', 'maelgwn', 
                'owain', 'kynon', 'kai', 'owain kynon', 'alma', 'benet', 'gurmukh', 'nanda', 'sagha', 'ambapl',
                'licchavis', 'bairaaree', 'raam', 'har har', 'kusinr', 'tathgata', 'sal', 'armf', 'orsha', 
                'odudwa', 'odwa', 'haungaroa', 'ueneku', 'potikiroroa', 'ku', 'heb', 'laban', 'hunahp', 'agusinnguaq',
                'knagssuaq', 'zipacn', 'cabracn', 'hunahp xbalanqu', 'xbalanqu', 'piqui', 'chaqui', 'piqui chaqui', 
                'coyllur', 'loki', 'idunn', 'troolie', 'luqman', 'tao', 'venddd', 'prabhaatee', 'basant', 'basant mehl', 
                'zoroaster', 'pahlavi', 'nanak', 'yasna', 'asha', 'baresman', 'master asha', 'malaar fifth', 'malaar',
                'maya', 'aasaa', 'intelezi', 'vara', 'goojaree', 'yima', 'uthlanga', 'utikxo', 'isalukazana', 'hrlfr', 
                'thjazi', 'ymir', 'kumagdlak', 'james', 'ornyan', 'bo', 'adaba', 'insingizi', 'hebus', 'mrimi', 'bo', 
                'moremi', 'ayar', 'viracocha', 'ccapac', 'manco', 'manco ccapac', 'shawano', 'gwyn', 'anarteq', 'kardltuarssuk', 
                'helen', 'vatea', 'llevelys', 'lludd', 'coranians', 'caridwen', 'einarr', 'einarr sang', 'gylfi', 
                'brokkr', 'owain', 'caribs', 'hreidmarr', 'freyr', 'whakaue', 'hotunui', 'wurrunnunnah', 'wirreenun', 
                'bunnyyarl', 'noondoo', 'byamee', 'ptussorssuaq', 'hariwali', 'asalq', 'arawaks', 'caribs', 'pomeroon', 
                'bootoolgah', 'goonur', 'comebee', 'bootoolgah goonur', 'corrobboree', 'ite', 'kurreahs', 'narran', 
                'byamee', 'haumea', 'tamure', 'kiki', 'olopana', 'goomblegubbon', 'goomblegubbons', 'irawaru', 'paka', 
                'kahureremoa', 'aotea', 'fakalan', 'kanawa', 'kapas', 'kgssagssuk', 'little kgssagssuk', 'umerdlugtoq', 
                'goonur', 'goonur husband', 'isigligrssik', 'chi', 'chi chi', 'birrahlee', 'wahroogahs', 'bunbundoolooey',
                'bangan', 'qujvrssuk', 'kapalama', 'amerdloq', 'simo', 'ebbong', 'corial', 'piai', 'gwineeboo', 'ah ah',
                'gidgereegah', 'aaron', 'amalekites', 'amulek', 'zarathushtra', 'korihor', 'adam', 'pharoah', 'iblis',
                'inkosazana', 'mormon', 'helaman', 'kishkumen', 'morianton', 'pahoran', 'amulek', 'zeezrom', 'judah', 
                'nips', 'cf', 'tamar', 'gond', 'mrimi', 'bo', 'shu', 'lamoni', 'cachi', 'kaw', 'narahdarn', 'surtr',
                'joseph', 'nanda', 'jee', 'yaya', 'brahmin', 'kaanraa', 'tilang', 'gurmukhs', 'benet', 'raamkalee',
                'mehl', 'baynee', 'lohicca', 'rosika', 'slavatik', 'rosika barber', 'saadh', 'saadh sangat',
                'fifth mehl', 'saarang fifth', 'saarang', 'fifth mehl', 'ephron', 'sarah', 'abimelech', 'freyja',
                'abram', 'sodom', 'zoar', 'iwa', 'umi', 'isigligrssik', 'galagnguas', 'rachel', 'leah', 'gooloo',
                'comebees', 'esau', 'rebekah', 'usigwili', 'amasi', 'umkqaekana', 'amazulu', 'armf', 'orsha', 'odudwa',
                'odwa', 'zipacn', 'cabracn', 'hunahp xbalanqu', 'xbalanqu', 'hunahp', 'mehl', 'inyanga', 'hai', 'impepo',
                'hai hai', 'oom', 'oom oom', 'wurrunnah', 'dayoorls', 'ukoko', 'ukulukulu', 'udhlamini', 'unsondo',
                'banab', 'maraka', 'kakuhihewa', 'gir', 'bragi', 'ubulawo', 'indras', 'shivas', 'mehl', 'second mehl',
                'uillac uma', 'uma', 'uillac', 'ravi', 'ravi daas', 'daas', 'bairaagi', 'soohee jee', 'nasu', 
                'balwand', 'khivi', 'ffnir', 'reginn', 'hgni', 'gudrn', 'jrmunrekkr', 'erpr', 'king jrmunrekkr', 'srli',
                'hamdir', 'kohora', 'moogaray', 'eehu', 'masilo', 'kenkebe', 'kardltuarssuk', 'raumati', 'karika',
                'kakei', 'tinirau', 'kae', 'tatau', 'amanxusa', 'rhiannon', 'pryderi rhiannon', 'pryderi', 'cantrevs',
                'ornyan', 'armf', 'nyanribo', 'yawarri', 'qujvrssuk', 'isokun', 'iddawc', 'beeargah', 'borah',
                'bendigeid', 'bendigeid vran', 'vran', 'branwen', 'matholwch', 'branwen', 'kae', 'tinirau', 'tutunui',
                'hrr', 'leto', 'genii', 'rua', 'tama', 'rata', 'kanaloa', 'aikanaka', 'kakuhihewa', 'kapoi', 'wabassi',
                'nuknguasik', 'kgssagssuk', 'little kgssagssuk', 'dkw', 'gatan', 'boliwan', 'ideo', 'anitos',
                'qujvrssuk', 'warribisi', 'moodai', 'paiwarri', 'kokerite', 'maikoha', 'ngatora', 'kahukura',
                'whatuiapiti', 'heimdallr', 'dionysus', 'makunaima', 'ornyan', 'bo', 'olbo', 'thjlfi', 'peredur',
                'mrimi', 'oluronbi', 'bamu', 'hioi', 'ptussorssuaq', 'yurokon', 'nat', 'nat fifth', 'dkw', 'kgssagssuk',
                'isigligrssik', 'hou', 'hawepotiki', 'uenuku', 'prahlaad', 'harnaakhash', 'ouyan', 'yuckay', 'yuckay yuckay',
                'comebo', 'cronos', 'ah', 'ah ah', 'ooboon', 'wh', 'whn', 'solomon', 'hunahp', 'mahthi', 'norns', 'hrr',
                'nk', 'njps', 'shechem', 'hamor', 'magahar', 'benares', 'bhairao', 'gwaarayree', 'waaho', 'hindol',
                'lord hindol', 've', 'nanda', 'sagha', 'tathgata', 'midgard', 'maxen', 'lamanites', 'zion', 'armf',
                'orsha', 'odwa', 'odudwa', 'zoramites', 'jershon', 'gadianton', 'jarom', 'lamanites', 'ammoron', 'antipus',
                'deepak', 'maalakausak', 'chaytee', 'lalo', 'gwydion', 'gronw', 'zion', 'umwathleni', 'zarahelma',
                'peredur', 'luned', 'yma sumac', 'yma', 'hrr', 'danom', 'gabi', 'gomotan', 'ponaturi', 
                'kgssagssuk', 'little kgssagssuk', 'takakopiri', 'dkw', 'oriyu', 'asalq', 'uktena', 
                'uktena', 'maipuri', 'wawaiya', 'tuwhakararo', 'warraus', 'kaupe', 'pwyll', 'teirnyon',
                'heveydd', 'gwawl', 'gwalchmai', 'geraint', 'dummerh', 'mooregoo', 'gwai', 'bilbers',
                'hauraki', 'whatu', 'ngatoro', 'soulbride', 'kavi', 'harimandir', 'noid', 'astivihad',
                'gazi', 'goolay', 'rhonabwy', 'armf', 'orsha', 'odudwa', 'odwa', 'gir', 'sita lachhman',
                'hrr', 'har', 'mabon', 'llew', 'son modron', 'modron', 'mabon son', 'hrr', 'anurruddha',
                'devats', 'aairs', 'aogemadaeca', 'kusinr', 'tathgata', 'baldr', 'svadilfari', 'sagha',
                'sagha monks', 'ambapl', 'ongkaar', 'tathgata', 'bairaagan', 'gideon', 'har', 
                'moronihah', 'zarahelma', 'sidon', 'king laman', 'amlicites',  'iaen', 'eri', 'greid', 'greid son',
                'shule', 'son eri', 'corihor', 'kib', 'akish', 'mazdayasnians', 'thravan', 'athravan', 'myazda', 'zaotar',
                'sherrizah', 'tet', 'middoni', 'benjamin', 'king benjamin', 'amulon', 'nemmes', 'xbalanqu', 'hunahp xbalanqu',
                'hunahp', 'zerahemnah', 'ornyan', 'bo', 'olbo', 'simbukumbukwana', 'mbulu', 'skrnir', 'frigg',
                'asalq', 'ffnir', 'qujvrssuk', 'geirrdr', 'grdr', 'erim', 'erim', 'twrch', 'gweir', 'boku',
                 'boku boku', 'severn', 'grugyn', 'jtunheim', 'annwvyn', 'prince dyved', 'ynwyl', 'etlym',
                 'kgssagssuk', 'little kgssagssuk', 'nahakoboni' 'qalagnguas', 'kardltuarssuk', 'ornyan',
                 'bo', 'olbo', 'armf', 'orsha', 'odwa', 'odudwa', 'pasnush', 'hana', 'zaurura', 'usithlanu',
                 'wadahans', 'manmukhs', 'yas', 'gthas', 'david', 'en', 'zipacn', 'toi', 'manawyddan',
                 'sty', 'powys', 'mathonwy', 'hrr', 'aairs', 'kusinr', 'tathgata', 'raag bihaagraa',
                 'bihaagraa', 'manmukh', 'aasaavaree', 'subhadda', 'nanda', 'ahurian', 'ganges', 'soohee',
                 'soohee fifth', 'sagha', 'ndik', 'nanda', 'igbos', 'mrimi', 'oranyan', 'isaiah', 'laman',
                 'knigseq', 'qalagnguas', 'isigligrssik', 'asalq', 'hatcinodo', 'ga na', 'ga', 'kgssagssuk',
                 'little kgssagssuk', 'qujvrssuk', 'pakuanui', 'huhuti', 'weeoombeen', 'wauke', 'popohorokewa',
                 'tamanoa', 'kaakau', 'llwyddeu', 'gwadyn', 'pehu', 'leho', 'gudrn', 'frdi', 'hgni', 'gjki',
                 'makanauro', 'trwyth', 'llwydawg', 'kalma', 'chhant', 'orm', 'azi', 'xix', 'mainyu',
                 'angra mainyu', 'angra', 'introd', 'yast', 'maalaa', 'maalaa fifth', 'kapo', 'kalihi',
                 'rhun', 'kaydaaraa', 'gwenhwyvar', 'edeyrn', 'enid', 'cusi', 'umdabuko', 'rened', 'nanda',
                 'tathgata', 'sagha', 'gwyddno', 'ishmael', 'gwenhwyvar', 'glewlwyd', 'ffnir', 'hermdr', 'hel',
                 'abinadi', 'naglfar', 'hrr', 'sidom', 'musan', 'meamei', 'qujvrssuk', 'hoahanau', 'ipukai',
                 'halemanu', 'saunikoq', 'zim', 'suttungr', 'ffnir', 'baugi', 'nongwes', 'magoda', 'dardurr',
                 'willgoo', 'evnissyen',  'lono', 'ano', 'scr', 'ii', 'tangaroa', 'umahaule', 'unqanqaza',
                 'matahorua', 'kuramarotini', 'amanthlwenga', 'umdhlebe', 'neruvkq', 'avvang', 'dyved',
                 'imamba', 'guluwe', 'hili', 'nand', 'raavan', 'wh', 'dayoorl', 'wh wh', 'whn', 'tangalimlibo',
                 'lehna', 'druj', 'cain', 'abel', 'armf', 'orsha', 'isigligrssik', 'qalagnguas', 'bilaawal',
                 'ood', 'mukanday', 'jrmunrekkr', 'frdi', 'omer', 'jvaka', 'inuence', 'sebus', 'melchizedek',
                 'aphrodite', 'anchises', 'konane', 'marabuntas', 'heiau', 'baddasan', 'celeus', 'cowee',
                 'hinai', 'pisi', 'hatcinodo', 'sgard', 'skrmir', 'nanyobo', 'cantrev', 'blodeuwedd',
                 'frdi', 'gwythyr', 'nudd', 'gwythyr son', 'angusinnguaq', 'dkw', 'vishtaspa', 'king vishtaspa',
                 'nabnazdistas', 'amrit', 'sita lachhman', 'lahore', 'lachhman', 'raag gujri', 'raag',
                 'yasht', 'drvaspa', 'kavis', 'undhlebekazizwa', 'umazwana', 'amakuza', 'ifr', 'uggason','ifr uggason',
                 'cabracn', 'madawc', 'tegid', 'bach', 'gwion bach', 'iorwerth', 'hunbatz', 'hunchoun', 'hunbatz hunchoun',
                 'mrimi', 'bo', 'hrlfr', 'edom', 'gujri fifth', 'gujri', 'kevaa', 'trilochan', 'veda', 'dakhmas',
                 'vi', 'tishtrya', 'goolahgool', 'angusinnguaq', 'kapa', 'wh', 'whn', 'bindeah', 'uhu',
                 'kauahoa', 'keaau', 'hauula', 'tarbaran', 'grdr', 'frdi', 'geirrdr', 'dungle', 'hrr',
                 'skrmir', 'knigseq', 'forth spitama', 'spitama', 'jahi', 'sannyaasi', 'blverkr', 'tr',
                 'hrr', 'mrimi', 'bo', 'ifes', 'apakura', 'rongotakawiu', 'dinah', 'quich', 'dhadha',
                 'devats', 'kusinr', 'jaijaavantee', 'jaijaavantee ninth', 'rened', 'jvaka', 'ajtasattu', 
                 'samiri', 'nahakoboni', 'waiamari', 'skdbladnir', 'ayo', 'maalee', 'gauraa', 'maalee gauraa',
                 'gauraa fifth', 'mithra', 'bhagaautee', 'vaishnaav', 'kalyaan', 'poorbee fourth',
                 'tamub', 'menw', 'ermid', 'shu', 'gunas', 'eing', 'gotama', 'sushmanaa', 'ambapl',
                 'nanda', 'cundal', 'vinaya', 'dakhanay fifth', 'lord dakhanay', 'dakhanay', 'maru', 'cumorah',
                 'helam', 'jvaka', 'ajtasattu', 'vassakra', 'amlici', 'aurvandill', 'arnrr', 'eilfr', 'zarahelma',
                 'erbin', 'rangitihi', 'tupenu', 'nuknguasik', 'hatcinodo', 'ch', 'hrlfr', 'kraki',
                 'hrlfr kraki', 'adils', 'gudrn', 'tane', 'zakariya', 'hgni', 'hedinn', 'nkws', 'phoebus',
                 'idzumo', 'brahm', 'benet', 'udaasee', 'nabnazdistas', 'seq seq', 'istrs', 'innite',
                 'yspaddaden penkawr', 'penkawr', 'yspaddaden', 'gwrnach', 'potoru', 'tuau', 'goug', 'gour gah', 'gour gour',
                 'gah gah', 'tahiti', 'ngahue', 'hawaiki', 'tainui', 'ihuatamai', 'hinauri' 'ihuwareware', 'yackman', 'whakaturia',
                 'amanna', 'artuk', 'ptussorssuaq', 'altaq', 'gwion', 'mawri', 'birnie', 'ullr', 'sark', 'rdi', 'hildr', 'hlkk',
                 'hildr', 'hgni', 'kvasir', 'skald', 'ifr', 'eyvindr', 'thrvaldi', 'hrr', 'whn', 'kalaniopuu', 'mayrah',
                 'idhlozi', 'wa', 'asalq', 'ngngjuk', 'nuknguasik', 'qalagnguas', 'isigligrssik', 'felin', 'punihuia', 'kushi',
                 'armorica', 'gillingr', 'blverkr', 'geirrdr', 'fedilizan', 'kilauea', 'seq', 'istrs', 'nabnazdistas',
                 'anquetil', 'saawan', 'irth', 'krishnas', 'vishtasp', 'visperad', 'vohu', 'rashnu', 'verethraghna'
                 'ersonified', 'ndying', 'ndying eyond', 'atred', 'eyond', 'reative', 'samaadhi', 'govind', 'haray', 'lord haray',
                 'haray haray', 'aweoweo', 'frdi', 'kuiwai', 'drengs', 'haraldr', 'cotuh', 'ahpop', 'nihaib', 'eepa', 'kearoa',
                 'mahina', 'waolani', 'kahiki', 'llyr', 'caradawc', 'harlech', 'kynan', 'reuel', 'chedorlaomer',
                 'dishon', 'basemath', 'oholibamah', 'zibeon', 'anah', 'eliphaz', 'eos', 'gandhaaree', 'raaginis', 'hagar',
                 'hera', 'orsha', 'brahm', 'great brahm', 'maghar', 'katak', 'phalgun', 'saawan', 'bhaadon', 'assu', 'maagh',
                 'ambapl', 'pv', 'cunda', 'vesl', 'arezra', 'brahmans', 'nite', 'innite', 'nite innite', 'augustness',
                 'salla', 'pitu', 'pitu salla', 'kusinr', 'nanda', 'lamech', 'nahor', 'teancum', 'haran', 'terah', 'seth',
                 'rangi', 'rangi papa', 'ahr', 'zarahelma', 'ynywl', 'azhi', 'dava', 'daeva', 'thravan', 'saddar hyde', 'saddar',
                 'ephraim', 'bethuel', 'lehi', 'baresma', 'utshaka', 'goolays', 'usenzangakona', 'uyegana', 'uxele', 'unsikana',
                 'ulangeni', 'umpengula', 'mbanda', 'umpengula mbanda', 'uthlomo', 'ammaron', 'senum', 'kamoiliili', 'arnrr',
                 'eilfr', 'boki', 'fridleifr', 'amalickiah', 'teancum', 'lehi', 'pershephone', 'sanjaya', 'drona', 'bhishma',
                 'skrmir', 'ukanzi', 'sgard', 'goore', 'goore goore', 'mirrieh', 'saxland', 'eudav', 'gathas', 'gir',
                 'polack', 'gomez', 'vanir', 'vaar', 'gursikhs', 'gorakh', 'puraanas', 'qazi', 'blerwm', 'yamato',
                 'naad', 'bragr', 'hrr', 'valhall', 'urdr', 'vlusp', 'dathyl', 'caer dathyl', 'custennin', 'hilu',
                 'bela', 'saul', 'cyllenian', 'makte', 'bedwyr', 'sarai', 'vag', 'airyana', 'airyana vag', 'frangrasyan',
                 'aryan', 'aredvi', 'airyaman', 'saoka', 'pali', 'ajtasattu', 'ajtasattu vedhiputta', 'vedehiputta',
                 'sunidha', 'sunidha vassakra', 'vassakra', 'aad', 'aad mam', 'mam tanvo', 'mam', 'ithyejanguhaiti',
                 'gtha', 'tanvo ithyejanguhaiti', 'tanvo', 'kusinr', 'mah kassapa', 'lakshmi', 'king ajtasattu',
                 'ajtasattu', 'nigaha', 'pendaran', 'dava', 'ashi', 'ashi vanguhi', 'vanguhi', 'dakhma', 'bheekhan', 'dwaarikaa',
                 'teomner', 'manti', 'lehonti', 'emer', 'thjlfi', 'grjtnagard', 'zemnarihah', 'mulek', 'geirrdr', 'arianrod',
                 'mspell', 'hestia', 'gatha', 'kinvad', 'gtha', 'karshvares', 'strabo', 'venddd', 'verethraghna', 'verethraghna verethraghna',
                 'mohinis', 'kahilona', 'gilvaethwy', 'gwynedd', 'gilvaethwy son', 'sijjin', 'kilitraq', 'hiiakas', 'leurs', 'gmz',
                 'comm', 'ne', 'draona', 'gahi', 'gucumatz', 'synjur', 'beli', 'hermdr', 'hunchoun', 'isiwandiye', 'hina',
                 'ndik', 'ew', 'aairs', 'pairika', 'keresspa', 'vedic', 'dahka', 'vedas', 'aminadab', 'mazdeism', 'king ajtasattu',
                 'kassapa', 'shiblom', 'yggdrasill', 'hrr', 'vlusp', 'bifrst', 'wh', 'whn', 'wh wh', 'wondah', 'amos',
                 'senine', 'ormazd', 'pitris', 'haurvatt amerett', 'baisakhi', 'hud', 'qibla', 'kaashi', 'grani', 'niflungs',
                 'hgni', 'fri', 'hjadnings', 'hrlfr', 'gjki', 'camlan', 'echel', 'son saidi', 'saidi', 'gwallt', 'refr', 'sif',
                 'blverkr', 'kamehameha', 'waikiki', 'waipio', 'ammonihah', 'melek', 'shiblon', 'sigyn', 'frbauti', 'jtunheim',
                 'thravan', 'havgan', 'arawn', 'kumaso', 'dhritirashtra', 'saadhus', 'vedas']
analyzer = ReligiousTextThemeAnalyzer(custom_stopwords=custom_stops)

# Load data
df = pd.read_csv('Dataset_with_Text.csv', encoding='utf-8')

# Analyze texts
results = analyzer.analyze_corpus(df)

#%%
# Perform topic similarity analysis
similarity_analyzer = TopicSimilarityAnalyzer(results, df)
topic_groups = similarity_analyzer.compute_topic_similarities(similarity_threshold=0.75)
similarity_analyzer.print_topic_groups(topic_groups)

# %%
print(df)
# %%
