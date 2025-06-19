To perform sentiment analysis from scratch in Python, you need to follow a series of steps that involve data processing, text representation, and applying a machine learning algorithm. Here’s a step-by-step guide:

### 1. **Data Collection**

You'll need a dataset to work with. A common dataset for sentiment analysis is the **IMDb movie reviews dataset** or **Twitter data** (positive or negative). For this example, let's assume you're working with a simple dataset of positive and negative sentences.

### 2. **Text Preprocessing**

The next step is to preprocess the text data. This involves:

* Lowercasing the text
* Removing punctuation
* Removing stopwords (common words like "and", "the", etc.)
* Tokenizing the text (breaking it into individual words)

#### Example:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "I love this movie! It's amazing."

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

# Preprocess the sample text
preprocessed_text = preprocess_text(text)
print(preprocessed_text)
```

### 3. **Creating a Bag of Words (BoW)**

In this step, you will convert the words into numerical features. One common approach is the **Bag of Words (BoW)** model, which represents each text as a vector of word counts.

#### Example:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample data (positive and negative sentences)
corpus = [
    "I love this movie, it was amazing",
    "Horrible movie, I hated it",
    "The movie was good but too long",
    "I really enjoyed the film",
    "This film was a waste of time"
]

# Initialize CountVectorizer (BoW)
vectorizer = CountVectorizer()

# Fit and transform the corpus to get word counts
X = vectorizer.fit_transform(corpus)

# Convert the result to an array and print it
print(X.toarray())
print(vectorizer.get_feature_names_out())
```

### 4. **Creating Labels (Sentiment Labels)**

You need to create labels for your dataset, where each text is classified as either "positive" or "negative".

```python
# Labels for the sample data (1 = Positive, 0 = Negative)
y = [1, 0, 1, 1, 0]  # Example labels
```

### 5. **Training a Model**

You can use a simple **Naive Bayes** classifier or a **Logistic Regression** model to train on the BoW representation of the text data.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### 6. **Making Predictions**

Now that you have a trained model, you can use it to predict the sentiment of new sentences.

```python
# New text to classify
new_text = ["I hated this movie, it was so boring"]

# Preprocess the new text
new_text_preprocessed = preprocess_text(new_text[0])

# Convert to a vector (BoW representation)
new_text_vectorized = vectorizer.transform([' '.join(new_text_preprocessed)])

# Predict sentiment
prediction = model.predict(new_text_vectorized)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

### 7. **Improving the Model (Optional)**

You can improve your sentiment analysis model by:

* Using **TF-IDF** instead of simple word counts for better word representation.
* Using **more complex models** like Support Vector Machines (SVM), or deep learning models like LSTM for better results.
* Collecting and annotating a larger, more diverse dataset.

---

### Complete Example:

Here’s the full code in one block:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample data
corpus = [
    "I love this movie, it was amazing",
    "Horrible movie, I hated it",
    "The movie was good but too long",
    "I really enjoyed the film",
    "This film was a waste of time"
]

# Labels for the sample data (1 = Positive, 0 = Negative)
y = [1, 0, 1, 1, 0]

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

# Preprocess the corpus
corpus_preprocessed = [' '.join(preprocess_text(text)) for text in corpus]

# Initialize CountVectorizer (BoW)
vectorizer = CountVectorizer()

# Fit and transform the corpus to get word counts
X = vectorizer.fit_transform(corpus_preprocessed)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# New text to classify
new_text = ["I hated this movie, it was so boring"]

# Preprocess the new text
new_text_preprocessed = preprocess_text(new_text[0])

# Convert to a vector (BoW representation)
new_text_vectorized = vectorizer.transform([' '.join(new_text_preprocessed)])

# Predict sentiment
prediction = model.predict(new_text_vectorized)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

This will get you up and running with a basic sentiment analysis model in Python. You can later expand on this by using more sophisticated models and fine-tuning for better accuracy.
