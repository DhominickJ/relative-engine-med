import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

class SymptomSimilarity: 
    def __init__(self, symptom_list):
        """ Initiatlize the symptom similarity model with a list of symptom."""
        self.symptom_list = symptom_list
        self.vectorizer = None
        self.symptom_vectors = None
    
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self._preprocess_symptoms()

    def _preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])

        # Remove stopwords and tokenize
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stopwords]

        # Lemmatization purposes
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(lemmatized)
    
    def _preprocess_symptoms(self):
        """Preprocess the symptoms and vectorize"""
        processed_symptoms = [self._preprocess_text(symptom) for symptom in self.symptom_list]

        # Create TF-IDF Vectors
        self.vectorizer = TfidfVectorizer()
        self.symptom_vectors = self.vectorizer.fit_transform(processed_symptoms)

    def find_similar_symptoms(self, input_text, top_n=10):
        """
        Find similar symptoms
        
        Args: 
            input_text: User input text
            top_n: Number of similar symptoms to return
            
        Returns:
            Lit of similar symptoms
        """

        processed_text = self._preprocess_text(input_text)
        
        input_vector = self.vectorizer.transform([processed_text])

        # Calculate similarity
        similarities = cosine_similarity(input_vector, self.symptom_vectors)[0]

        #Get top similar symptoms
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_symptoms = [(self.symptom_list[i], similarities[i]) for i in top_indices if similarities[i] > 0]

        return top_symptoms

    def get_similar_symptoms(self, symptom, top_n=5):
        """Find symptoms similar to a given symptom.
        
        Args: 
            symptom: Target symptom
            top_n: Number of similar symptoms to return
            
        Returns:
            List of top symptom based on top_n
            
        """

        try:
            symptom_idx = self.symptom_list.index(symptom)

            #get the vector for the target symptom
            target_vector = self.symptom_vectors[symptom_idx]

            #Calculate similarity
            similarities = cosine_similarity(target_vector, self.symptom_vectors)[0]

            # Get top symptoms
            top_indices = np.argsort(similarities)[::-1]

            # Filter out the target symptom and get top n
            similar_symptoms = []
            for idx in top_indices:
                if self.symptom_list[idx] != symptom and similarities[idx] > 0:
                    similar_symptoms.append((self.symptom_list[idx], similarities[idx]))
                    if len(similar_symptoms) >= top_n:
                        break
            
            return similar_symptoms
        except ValueError:
            return []  