# 📱 SMS Spam Detection Model

This project is a machine learning model developed using the **Scikit-learn** library that classifies English SMS messages as either **spam** or **ham** (non-spam).

> The model was trained on the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## ✨ Key Features

- **High Accuracy:** Achieves around **97% accuracy** on the test dataset.  
- **User-Friendly:** Has **0 false positives**, meaning no normal message is misclassified as spam — minimizing the risk of important messages being flagged incorrectly.  
- **Lightweight & Fast:** Model files are only a few megabytes, and predictions are made in milliseconds — ideal for real-time applications.  
- **Easy Integration:** Can be easily integrated into any Python project.

---

## 📂 Project Structure

```
.
├── spam_model.pkl          # Trained Naive Bayes model
├── vectorizer.pkl          # Trained TF-IDF vectorizer
├── spam.csv 
├── Spam_SMS_Collection.ipynb  # Jupyter Notebook for model training and development
└── README.md               
```


---

## 🚀 Installation & Setup

Follow these steps to run the project locally:

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)[USERNAME]/[REPO_NAME].git
cd [REPO_NAME]

```

**2.Install the required libraries:
Use the requirements.txt file to install dependencies:
```bash
pip install -r requirements.txt
```
If you don’t have a requirements.txt file, you can manually install the libraries:
```bash
pip install pandas numpy scikit-learn nltk
```

**3.Download NLTK Stopwords:
The model requires NLTK’s stopwords list for text preprocessing. Run the following commands in a Python interpreter:
```python
import nltk
nltk.download('stopwords')
```

## 💻 Example Usage
You can easily use the trained model in your own Python project.
Example (example.py):

```python
# example.py
import pickle

def predict_spam(message: str) -> str:
    """
    Predicts whether a given message is spam or not.
    """
    try:
        # Load the saved vectorizer and model
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return "Error: Model files ('vectorizer.pkl', 'spam_model.pkl') not found."

    # Transform the input message using the vectorizer
    message_tfidf = vectorizer.transform([message])
    
    # Make prediction
    prediction = model.predict(message_tfidf)
    
    return prediction[0].upper()

# --- Test ---
spam_message = "Congratulations! You've won a $1000 Walmart gift card. Go to http://example.com to claim now."
normal_message = "Hi mom, I'll be home late for dinner tonight. Can you save me some food?"

print(f"Message: '{spam_message}'")
print(f"  -> Prediction: {predict_spam(spam_message)}\n")

print(f"Message: '{normal_message}'")
print(f"  -> Prediction: {predict_spam(normal_message)}")

```

## 📊 Model Performance
The model was evaluated on 20% of the dataset (test set).
Here are the results:

| Metric                           | Value  |
| :------------------------------- | :----- |
| **Accuracy**                     | 96.77% |
| **Ham Precision**                | 96%    |
| **Spam Precision**               | 100%   |
| **False Positives (Ham → Spam)** | **0**  |



## ⚠️ Limitations
Language: Designed exclusively for English text messages.
Format: Works best with short texts such as SMS. May perform less effectively on long emails or other formats.
Usage Purpose: It is recommended to flag messages as “possible spam” instead of deleting them automatically.

## 📄 License
This project is licensed under the Apache 2.0 License.
See the LICENSE file for more details.

...by Vira
