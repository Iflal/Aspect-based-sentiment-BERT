from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
import re

class AspectSentimentModel:
    def __init__(self, model_dir="saved_sentiment_model"):
        # Path to the saved model
        self.model_dir = model_dir
        
        # Load the model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        # Initialize the sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, truncation=True)
        
        # Define the aspect keywords (can be predefined in the model itself)
        self.aspect_keywords = {
            'Food Quality': ['food', 'taste', 'flavor', 'meal', 'dish', 'cuisine', 'delicious', 'bland'],
            'Service': ['service', 'staff', 'waiter', 'waitress', 'attentive', 'slow', 'friendly', 'rude'],
            'Ambiance': ['ambiance', 'atmosphere', 'environment', 'decor', 'vibe', 'music', 'noisy', 'quiet']
        }
    
    def extract_aspect_text(self, review, aspect_name):
        """ Extract aspect-related text from review based on predefined keywords. """
        keywords = self.aspect_keywords.get(aspect_name, [])
        sentences = re.split(r'[.!?]', review)  # Split review into sentences
        aspect_text = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
        return ' '.join(aspect_text)
    
    def analyze_sentiment(self, aspect_text):
        """ Analyze sentiment for the aspect using the pre-trained model. """
        if aspect_text.strip():
            result = self.sentiment_analyzer(aspect_text)[0]
            return result['label'].lower()
        return 'neutral'

    def analyze_review(self, review):
        """ Analyze sentiment for multiple aspects in a review. """
        result = {}
        
        # Extract and analyze sentiment for each aspect
        for aspect_name in self.aspect_keywords:
            aspect_text = self.extract_aspect_text(review, aspect_name)
            sentiment = self.analyze_sentiment(aspect_text)
            result[aspect_name + " Sentiment"] = sentiment
        
        return result

# # Save model and tokenizer (in case you haven't already)
# def save_model():
#     model_dir = "saved_sentiment_model"
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)
#     sentiment_analyzer.model.save_pretrained(model_dir)
#     sentiment_analyzer.tokenizer.save_pretrained(model_dir)
#     print(f"Model and tokenizer saved in '{model_dir}'.")

# Load the model and analyze a new review
if __name__ == "__main__":
    # Load the pre-trained model
    model = AspectSentimentModel(model_dir="saved_sentiment_model")
    
    # Example review
    new_review = "The pasta was absolutely delicious, perfectly cooked with a great blend of spices. However, the service was disappointing; the waiter took too long to attend to us, and we had to wait ages for our drinks. The ambiance was charming and cozy, with beautiful decor and soft lighting."
    
    # Get the sentiment analysis result
    result = model.analyze_review(new_review)
    
    # Print the result
    print(result)
