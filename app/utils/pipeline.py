"""
Toxicity Classifier Pipeline

A comprehensive text classification system for detecting toxic content
across multiple categories using ensemble logistic regression models.
Based on the research notebook approach with balanced datasets.
"""

import re
import string
import pickle
import joblib
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ToxicityTextProcessor:
    """
    Text preprocessing component for toxicity classification.
    Handles all text cleaning, normalization, and feature extraction.
    Based on the preprocessing pipeline from the research notebook.
    """
    
    # Class constants for abbreviations and contractions
    ABBREVIATIONS = {
        "$": " dollar ",
        "€": " euro ",
        "4ao": "for adults only",
        "a.m": "before midday",
        "a3": "anytime anywhere anyplace",
        "aamof": "as a matter of fact",
        "acct": "account",
        "adih": "another day in hell",
        "afaic": "as far as i am concerned",
        "afaict": "as far as i can tell",
        "afaik": "as far as i know",
        "afair": "as far as i remember",
        "afk": "away from keyboard",
        "app": "application",
        "approx": "approximately",
        "apps": "applications",
        "asap": "as soon as possible",
        "asl": "age, sex, location",
        "atk": "at the keyboard",
        "ave.": "avenue",
        "aymm": "are you my mother",
        "ayor": "at your own risk",
        "b&b": "bed and breakfast",
        "b+b": "bed and breakfast",
        "b.c": "before christ",
        "b2b": "business to business",
        "b2c": "business to customer",
        "b4": "before",
        "b4n": "bye for now",
        "b@u": "back at you",
        "bae": "before anyone else",
        "bak": "back at keyboard",
        "bbbg": "bye bye be good",
        "bbc": "british broadcasting corporation",
        "bbias": "be back in a second",
        "bbl": "be back later",
        "bbs": "be back soon",
        "be4": "before",
        "bfn": "bye for now",
        "blvd": "boulevard",
        "bout": "about",
        "brb": "be right back",
        "bros": "brothers",
        "brt": "be right there",
        "bsaaw": "big smile and a wink",
        "btw": "by the way",
        "bwl": "bursting with laughter",
        "c/o": "care of",
        "cet": "central european time",
        "cf": "compare",
        "cia": "central intelligence agency",
        "csl": "can not stop laughing",
        "cu": "see you",
        "cul8r": "see you later",
        "cv": "curriculum vitae",
        "cwot": "complete waste of time",
        "cya": "see you",
        "cyt": "see you tomorrow",
        "dae": "does anyone else",
        "dbmib": "do not bother me i am busy",
        "diy": "do it yourself",
        "dm": "direct message",
        "dwh": "during work hours",
        "e123": "easy as one two three",
        "eet": "eastern european time",
        "eg": "example",
        "embm": "early morning business meeting",
        "encl": "enclosed",
        "encl.": "enclosed",
        "etc": "and so on",
        "faq": "frequently asked questions",
        "fawc": "for anyone who cares",
        "fb": "facebook",
        "fc": "fingers crossed",
        "fig": "figure",
        "fimh": "forever in my heart",
        "ft.": "feet",
        "ft": "featuring",
        "ftl": "for the loss",
        "ftw": "for the win",
        "fwiw": "for what it is worth",
        "fyi": "for your information",
        "g9": "genius",
        "gahoy": "get a hold of yourself",
        "gal": "get a life",
        "gcse": "general certificate of secondary education",
        "gfn": "gone for now",
        "gg": "good game",
        "gl": "good luck",
        "glhf": "good luck have fun",
        "gmt": "greenwich mean time",
        "gmta": "great minds think alike",
        "gn": "good night",
        "g.o.a.t": "greatest of all time",
        "goat": "greatest of all time",
        "goi": "get over it",
        "gps": "global positioning system",
        "gr8": "great",
        "gratz": "congratulations",
        "gyal": "girl",
        "h&c": "hot and cold",
        "hp": "horsepower",
        "hr": "hour",
        "hrh": "his royal highness",
        "ht": "height",
        "ibrb": "i will be right back",
        "ic": "i see",
        "icq": "i seek you",
        "icymi": "in case you missed it",
        "idc": "i do not care",
        "idgadf": "i do not give a damn fuck",
        "idgaf": "i do not give a fuck",
        "idk": "i do not know",
        "ie": "that is",
        "i.e": "that is",
        "ifyp": "i feel your pain",
        "IG": "instagram",
        "iirc": "if i remember correctly",
        "ilu": "i love you",
        "ily": "i love you",
        "imho": "in my humble opinion",
        "imo": "in my opinion",
        "imu": "i miss you",
        "iow": "in other words",
        "irl": "in real life",
        "j4f": "just for fun",
        "jic": "just in case",
        "jk": "just kidding",
        "jsyk": "just so you know",
        "l8r": "later",
        "lb": "pound",
        "lbs": "pounds",
        "ldr": "long distance relationship",
        "lmao": "laugh my ass off",
        "lmfao": "laugh my fucking ass off",
        "lol": "laughing out loud",
        "ltd": "limited",
        "ltns": "long time no see",
        "m8": "mate",
        "mf": "motherfucker",
        "mfs": "motherfuckers",
        "mfw": "my face when",
        "mofo": "motherfucker",
        "mph": "miles per hour",
        "mr": "mister",
        "mrw": "my reaction when",
        "ms": "miss",
        "mte": "my thoughts exactly",
        "nagi": "not a good idea",
        "nbc": "national broadcasting company",
        "nbd": "not big deal",
        "nfs": "not for sale",
        "ngl": "not going to lie",
        "nhs": "national health service",
        "nrn": "no reply necessary",
        "nsfl": "not safe for life",
        "nsfw": "not safe for work",
        "nth": "nice to have",
        "nvr": "never",
        "nyc": "new york city",
        "oc": "original content",
        "og": "original",
        "ohp": "overhead projector",
        "oic": "oh i see",
        "omdb": "over my dead body",
        "omg": "oh my god",
        "omw": "on my way",
        "p.a": "per annum",
        "p.m": "after midday",
        "pm": "prime minister",
        "poc": "people of color",
        "pov": "point of view",
        "pp": "pages",
        "ppl": "people",
        "prw": "parents are watching",
        "ps": "postscript",
        "pt": "point",
        "ptb": "please text back",
        "pto": "please turn over",
        "qpsa": "what happens",
        "ratchet": "rude",
        "rbtl": "read between the lines",
        "rlrt": "real life retweet",
        "rofl": "rolling on the floor laughing",
        "roflol": "rolling on the floor laughing out loud",
        "rotflmao": "rolling on the floor laughing my ass off",
        "rt": "retweet",
        "ruok": "are you ok",
        "sfw": "safe for work",
        "sk8": "skate",
        "smh": "shake my head",
        "sq": "square",
        "srsly": "seriously",
        "ssdd": "same stuff different day",
        "tbh": "to be honest",
        "tbs": "tablespooful",
        "tbsp": "tablespooful",
        "tfw": "that feeling when",
        "thks": "thank you",
        "tho": "though",
        "thx": "thank you",
        "tia": "thanks in advance",
        "til": "today i learned",
        "tl;dr": "too long i did not read",
        "tldr": "too long i did not read",
        "tmb": "tweet me back",
        "tntl": "trying not to laugh",
        "ttyl": "talk to you later",
        "u": "you",
        "u2": "you too",
        "u4e": "yours for ever",
        "utc": "coordinated universal time",
        "w/": "with",
        "w/o": "without",
        "w8": "wait",
        "wassup": "what is up",
        "wb": "welcome back",
        "wtf": "what the fuck",
        "wtg": "way to go",
        "wtpa": "where the party at",
        "wuf": "where are you from",
        "wuzup": "what is up",
        "wywh": "wish you were here",
        "yd": "yard",
        "ygtr": "you got that right",
        "ynk": "you never know",
        "zzz": "sleeping bored and tired"
    }
    
    def __init__(self) -> None:
        """Initialize the text processor with NLTK components."""
        self._initialize_nltk_components()
    
    def _initialize_nltk_components(self) -> None:
        """Initialize NLTK tokenizer and download required resources."""
        try:
            nltk.download('stopwords', quiet=True)
            self.tokenizer = TweetTokenizer(strip_handles=True)
            self.stop_words = set(stopwords.words('english'))
            self.corpus = []  # For compatibility with notebook code
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NLTK components: {e}")
    
    def clean(self, tweet: str) -> str:
        """
        Clean text using the exact methodology from the research notebook.
        This replicates the clean() function from the notebook.
        """
        # Contractions - exact patterns from notebook
        contractions = [
            (r"he's", "he is"), (r"there's", "there is"), (r"We're", "We are"),
            (r"That's", "That is"), (r"won't", "will not"), (r"they're", "they are"),
            (r"Can't", "Cannot"), (r"wasn't", "was not"), (r"don\x89Ûªt", "do not"),
            (r"aren't", "are not"), (r"isn't", "is not"), (r"What's", "What is"),
            (r"haven't", "have not"), (r"hasn't", "has not"), (r"There's", "There is"),
            (r"He's", "He is"), (r"It's", "It is"), (r"You're", "You are"),
            (r"I'M", "I am"), (r"shouldn't", "should not"), (r"wouldn't", "would not"),
            (r"i'm", "I am"), (r"I\x89Ûªm", "I am"), (r"I'm", "I am"),
            (r"Isn't", "is not"), (r"Here's", "Here is"), (r"you've", "you have"),
            (r"you\x89Ûªve", "you have"), (r"we're", "we are"), (r"what's", "what is"),
            (r"couldn't", "could not"), (r"we've", "we have"), (r"it\x89Ûªs", "it is"),
            (r"doesn\x89Ûªt", "does not"), (r"It\x89Ûªs", "It is"), (r"Here\x89Ûªs", "Here is"),
            (r"who's", "who is"), (r"I\x89Ûªve", "I have"), (r"y'all", "you all"),
            (r"can\x89Ûªt", "cannot"), (r"would've", "would have"), (r"it'll", "it will"),
            (r"we'll", "we will"), (r"wouldn\x89Ûªt", "would not"), (r"We've", "We have"),
            (r"he'll", "he will"), (r"Y'all", "You all"), (r"Weren't", "Were not"),
            (r"Didn't", "Did not"), (r"they'll", "they will"), (r"they'd", "they would"),
            (r"DON'T", "DO NOT"), (r"That\x89Ûªs", "That is"), (r"they've", "they have"),
            (r"i'd", "I would"), (r"should've", "should have"), (r"You\x89Ûªre", "You are"),
            (r"where's", "where is"), (r"Don\x89Ûªt", "Do not"), (r"we'd", "we would"),
            (r"i'll", "I will"), (r"weren't", "were not"), (r"They're", "They are"),
            (r"Can\x89Ûªt", "Cannot"), (r"you\x89Ûªll", "you will"), (r"I\x89Ûªd", "I would"),
            (r"let's", "let us"), (r"it's", "it is"), (r"can't", "cannot"),
            (r"don't", "do not"), (r"you're", "you are"), (r"i've", "I have"),
            (r"that's", "that is"), (r"i'll", "I will"), (r"doesn't", "does not"),
            (r"i'd", "I would"), (r"didn't", "did not"), (r"ain't", "am not"),
            (r"you'll", "you will"), (r"I've", "I have"), (r"Don't", "do not"),
            (r"I'll", "I will"), (r"I'd", "I would"), (r"Let's", "Let us"),
            (r"you'd", "You would"), (r"It's", "It is"), (r"Ain't", "am not"),
            (r"Haven't", "Have not"), (r"Could've", "Could have"), (r"youve", "you have"),
            (r"donå«t", "do not")
        ]
        
        for pattern, replacement in contractions:
            tweet = re.sub(pattern, replacement, tweet)
        
        # Informal expressions
        informal_replacements = [
            (r"some1", "someone"), (r"yrs", "years"), (r"hrs", "hours"),
            (r"2morow|2moro", "tomorrow"), (r"2day", "today"), (r"4got|4gotten", "forget"),
            (r"b-day|bday", "b-day"), (r"mother's", "mother"), (r"mom's", "mom"),
            (r"dad's", "dad"), (r"hahah|hahaha|hahahaha", "haha"), (r"lmao|lolz|rofl", "lol"),
            (r"thanx|thnx", "thanks"), (r"goood", "good")
        ]
        
        for pattern, replacement in informal_replacements:
            tweet = re.sub(pattern, replacement, tweet)
        
        # Character entity references
        tweet = re.sub(r"&gt;", ">", tweet)
        tweet = re.sub(r"&lt;", "<", tweet)
        tweet = re.sub(r"&amp;", "&", tweet)
        
        # Typos, slang and informal abbreviations
        tweet = re.sub(r"w/e", "whatever", tweet)
        tweet = re.sub(r"w/", "with", tweet)
        tweet = re.sub(r"<3", "love", tweet)
        
        # URLs
        tweet = re.sub(r"http\S+", "", tweet)
        
        # Numbers
        tweet = re.sub(r'[0-9]', '', tweet)
        
        # Eliminating the mentions
        tweet = re.sub("(@[A-Za-z0-9_]+)", "", tweet)
        
        # Remove punctuation and special chars (keep '!')
        for p in string.punctuation.replace('!', ''):
            tweet = tweet.replace(p, '')
        
        # ... and ..
        tweet = tweet.replace('...', ' ... ')
        if '...' not in tweet:
            tweet = tweet.replace('..', ' ... ')
        
        # Tokenize
        tweet_words = self.tokenizer.tokenize(tweet)
        
        # Eliminating the word if its length is less than 3
        tweet = [w for w in tweet_words if len(w) > 2]
        
        # remove stopwords
        tweet = [w.lower() for w in tweet if w not in self.stop_words]
        
        self.corpus.append(tweet)
        
        # join back
        tweet = ' '.join(tweet)
        
        return tweet
    
    def convert_abbrev_in_text(self, tweet: str) -> str:
        """
        Convert abbreviations using the exact methodology from the research notebook.
        This replicates the convert_abbrev_in_text() function from the notebook.
        """
        t = []
        words = tweet.split()
        t = [self.ABBREVIATIONS[w.lower()] if w.lower() in self.ABBREVIATIONS.keys() else w for w in words]
        return ' '.join(t)
    
    def prepare_string(self, tweet: str, max_length: Optional[int] = 30) -> str:
        """
        Complete text preprocessing pipeline.
        This replicates the prepare_string() function from the notebook.
        
        Args:
            tweet: Input text
            max_length: Optional maximum character length before processing
        """
        # Optional length limiting
        if max_length and len(tweet) > max_length:
            tweet = tweet[:max_length]
        
        tweet = self.clean(tweet)
        tweet = self.convert_abbrev_in_text(tweet)
        return tweet
    
    def prepare_input(self, text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
        """
        Prepare input for model prediction.
        This replicates the prepare_input() function from the notebook.
        """
        text = self.prepare_string(text)
        text = vectorizer.transform([text])  # Keep as sparse matrix
        return text


class ToxicityClassifierInference:
    """
    Inference-only toxicity classifier for production use.
    Loads pre-trained models and provides fast predictions.
    Based on the logistic regression ensemble approach from the research notebook.
    """
    
    TOXICITY_CLASSES = [
        'toxic', 'severe_toxic', 'obscene', 
        'threat', 'insult', 'identity_hate'
    ]
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the inference classifier.
        
        Args:
            model_path: Path to the saved model directory or file
        """
        self.classes = self.TOXICITY_CLASSES.copy()
        self.text_processor = ToxicityTextProcessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.models: List[LogisticRegression] = []
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained model from disk.
        
        Args:
            model_path: Path to the saved model directory or file
        """
        model_path = Path(model_path)
        
        try:
            if model_path.is_dir():
                # Load from directory structure
                self._load_from_directory(model_path)
            else:
                # Load from single pickle file
                self._load_from_pickle(model_path)
            
            self.is_loaded = True
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def _load_from_directory(self, model_dir: Path) -> None:
        """Load model components from directory structure."""
        # Load vectorizer
        vectorizer_path = model_dir / "vectorizer.pkl"
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load individual models
        self.models = []
        for class_name in self.classes:
            model_path = model_dir / f"model_{class_name}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model for {class_name} not found at {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.models.append(model)
    
    def _load_from_pickle(self, pickle_path: Path) -> None:
        """Load complete model from single pickle file."""
        with open(pickle_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.models = model_data['models']
        
        # Validate loaded data
        if len(self.models) != len(self.classes):
            raise ValueError(f"Expected {len(self.classes)} models, got {len(self.models)}")
    
    def predict_probabilities(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predict toxicity probabilities for input text(s).
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Array of shape (n_samples, n_classes) with probability predictions
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        # Handle single string input
        if isinstance(text, str):
            text = [text]
        
        # Preprocess texts
        processed_texts = [self.text_processor.prepare_string(t) for t in text]
        
        # Vectorize - keep as sparse matrix
        X = self.vectorizer.transform(processed_texts)
        
        # Get predictions from each model
        predictions = np.zeros((len(text), len(self.classes)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        return predictions
    
    def predict(
        self, 
        text: Union[str, List[str]], 
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary toxicity classifications.
        
        Args:
            text: Single text string or list of texts
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Binary predictions array of shape (n_samples, n_classes)
        """
        probabilities = self.predict_probabilities(text)
        return (probabilities > threshold).astype(int)
    
    def predict_single(self, text: str, return_probabilities: bool = True) -> Dict[str, float]:
        """
        Get prediction results for a single text.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return probabilities or binary predictions
            
        Returns:
            Dictionary mapping class names to scores
        """
        if return_probabilities:
            probabilities = self.predict_probabilities(text)[0]
            return dict(zip(self.classes, probabilities))
        else:
            predictions = self.predict(text)[0]
            return dict(zip(self.classes, predictions))
        
    def predict_toxic(
        self, 
        text: str, 
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Replicate the predict_toxic function from the research notebook.
        
        Args:
            text: Input text to classify
            verbose: Whether to print results
            
        Returns:
            Dictionary with prediction results
        """
        if verbose:
            print(f"Raw text : {text}\n")
        
        # Prepare input using the same method as notebook
        processed_text = self.text_processor.prepare_string(text)
        vectorized_text = self.vectorizer.transform([processed_text])  # Keep sparse
        
        if verbose:
            print(f"Prepared text : {processed_text}\n")
        
        # Get predictions from each model
        results = {}
        for i, class_name in enumerate(self.classes):
            prediction = self.models[i].predict(vectorized_text)[0]
            probability = self.models[i].predict_proba(vectorized_text)[0][1]
            
            results[class_name] = {
                'prediction': prediction,
                'probability': probability
            }
            
            if verbose:
                print(f"{class_name} : {np.round(prediction, 4) * 100} %")
        
        if verbose:
            print("\n")
        
        return {
            'raw_text': text,
            'processed_text': processed_text,
            'predictions': results,
            'is_toxic': any(results[cls]['prediction'] for cls in self.classes),
            'max_toxicity_class': max(results, key=lambda x: results[x]['probability']),
            'max_toxicity_score': max(results[cls]['probability'] for cls in self.classes)
        }


class ToxicityClassifierTrainer:
    """
    Training class for the toxicity classifier.
    Based on the balanced dataset + logistic regression approach from the research notebook.
    """
    
    TOXICITY_CLASSES = [
        'toxic', 'severe_toxic', 'obscene', 
        'threat', 'insult', 'identity_hate'
    ]
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the trainer."""
        self.classes = self.TOXICITY_CLASSES.copy()
        self.text_processor = ToxicityTextProcessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.models: List[LogisticRegression] = []
        self.random_state = random_state
    
    def train_logistic_regression_on_1_class(
        self, 
        data: pd.DataFrame, 
        class_targeted: str, 
        vectorizer: TfidfVectorizer
    ) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
        """
        Train a logistic regression model for a single class using balanced dataset.
        This replicates the train_logistic_regression_on_1_class function from the notebook.
        
        Args:
            data: Training dataframe
            class_targeted: Target class name
            vectorizer: Fitted TF-IDF vectorizer
            
        Returns:
            Tuple of (model, X_test, y_test)
        """
        # Separate positive and negative samples
        positive_samples = data[data[class_targeted] == 1]
        negative_samples = data[data[class_targeted] == 0]
        
        # Sample the same number of negative samples as positive samples
        num_positive = len(positive_samples)
        print(f"Training {class_targeted}: {num_positive} positive samples")
        
        if num_positive == 0:
            raise ValueError(f"No positive samples found for class {class_targeted}")
        
        sampled_negative_samples = negative_samples.sample(
            n=min(num_positive, len(negative_samples)), 
            random_state=self.random_state
        )
        
        # Combine positive samples with sampled negative samples
        balanced_df = pd.concat([positive_samples, sampled_negative_samples])
        
        # Shuffle the balanced DataFrame
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Keep as sparse matrix - DON'T convert to dense array
        X = vectorizer.transform(balanced_df['comment_text'])
        y = balanced_df[[class_targeted]].values.ravel()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.025, random_state=self.random_state
        )
        
        # Logistic regression works with sparse matrices
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def fit(self, data_path: str = None, data: pd.DataFrame = None) -> 'ToxicityClassifierTrainer':
        """
        Train the toxicity classifier on provided data.
        Based on the training pipeline from the research notebook.
        
        Args:
            data_path: Path to training CSV file (default: 'data/train.csv')
            data: Pre-loaded dataframe (alternative to data_path)
        """
        # Load data
        if data is not None:
            df = data.copy()
        elif data_path:
            df = pd.read_csv(data_path)
        else:
            # Default path based on your project structure
            default_path = Path(__file__).parent.parent / "data" / "train.csv"
            if not default_path.exists():
                raise FileNotFoundError(f"Training data not found at {default_path}")
            df = pd.read_csv(default_path)
        
        # Validate required columns
        required_columns = ['comment_text'] + self.classes
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Drop ID column if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        
        print("Preprocessing text data...")
        # Apply text preprocessing using the same method as notebook
        df['comment_text'] = df['comment_text'].apply(self.text_processor.prepare_string)
        
        print("Fitting text vectorizer...")
        # Optimize TF-IDF vectorizer to reduce memory usage
        self.vectorizer = TfidfVectorizer(
            max_features=50000,  # Limit vocabulary size
            min_df=2,           # Ignore terms that appear in less than 2 documents
            max_df=0.95,        # Ignore terms that appear in more than 95% of documents
            ngram_range=(1, 2), # Use unigrams and bigrams
            stop_words='english'
        )
        self.vectorizer.fit(df['comment_text'])
        
        print(f"Vectorizer vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        print("Training individual classifiers...")
        # Train individual classifiers using the notebook approach
        self.models = []
        for class_name in self.classes:
            print(f"  Training classifier for {class_name}...")
            model, _, _ = self.train_logistic_regression_on_1_class(df, class_name, self.vectorizer)
            self.models.append(model)
        
        print("Training completed!")
        return self
    
    def save_model(self, save_path: str, save_format: str = 'pickle') -> None:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path where to save the model
            save_format: 'directory' or 'pickle' format
        """
        if not self.vectorizer or not self.models:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(save_path)
        
        if save_format == 'directory':
            self._save_to_directory(save_path)
        elif save_format == 'pickle':
            self._save_to_pickle(save_path)
        else:
            raise ValueError(f"Invalid save format: {save_format}")
        
        print(f"Model saved successfully to {save_path}")
    
    def _save_to_directory(self, save_dir: Path) -> None:
        """Save model components to directory structure."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        vectorizer_path = save_dir / "vectorizer.pkl"
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save individual models
        for i, class_name in enumerate(self.classes):
            model_path = save_dir / f"model_{class_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[i], f)
        
        # Save metadata
        metadata = {
            'classes': self.classes,
            'random_state': self.random_state,
            'model_count': len(self.models)
        }
        
        metadata_path = save_dir / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_to_pickle(self, pickle_path: Path) -> None:
        """Save complete model to single pickle file."""
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'classes': self.classes,
            'random_state': self.random_state
        }
        
        # Ensure directory exists
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)


# Convenience functions for quick usage
def load_toxicity_classifier(model_path: str) -> ToxicityClassifierInference:
    """
    Convenience function to load a trained toxicity classifier.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded classifier ready for inference
    """
    return ToxicityClassifierInference(model_path)


def train_new_model(
    data_path: str = None, 
    save_path: str = "models/toxicity_model.pkl"
) -> ToxicityClassifierInference:
    """
    Convenience function to train a new model and return inference classifier.
    
    Args:
        data_path: Path to training data CSV
        save_path: Where to save the trained model
        
    Returns:
        Trained classifier ready for inference
    """
    trainer = ToxicityClassifierTrainer()
    trainer.fit(data_path=data_path)
    trainer.save_model(save_path)
    
    return load_toxicity_classifier(save_path)


# Example usage and testing
if __name__ == "__main__":
    print("=== Toxicity Classification Pipeline - Training Mode ===")
    print("Training new toxicity classifier model...")
    yes = True
    if yes:
        print("\n=== Quick Validation Test ===")
        model_save_path = "models/toxicity_model.pkl"
        classifier = ToxicityClassifierInference(model_save_path)
        test_result = classifier.predict_single("Youre a bitch", return_probabilities=True)
        print("Test prediction successful - model is working correctly!")
        print(f"Test result: {test_result}")
    else:
        try:
            # Initialize trainer
            trainer = ToxicityClassifierTrainer()
            
            # Train the model using default data path (data/train.csv)
            print("Starting training process...")
            trainer.fit()
            
            # Save the trained model
            model_save_path = "models/toxicity_model.pkl"
            print(f"Saving model to {model_save_path}...")
            trainer.save_model(model_save_path)
            
            print("✓ Model training and saving completed successfully!")
            print(f"✓ Model saved at: {model_save_path}")
            print("✓ You can now use this model for inference in your Streamlit app")
            
            # Optional: Quick validation test
            print("\n=== Quick Validation Test ===")
            classifier = ToxicityClassifierInference(model_save_path)
            test_result = classifier.predict_single("This is a test message", return_probabilities=True)
            print("Test prediction successful - model is working correctly!")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Training data not found - {e}")
            print("Please ensure train.csv is located at 'data/train.csv'")
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("Please check your data format and try again")