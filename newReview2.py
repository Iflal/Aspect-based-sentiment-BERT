#import a AspectSentimentModel class from newReview.py
from newReview import AspectSentimentModel


# Create an instance of the AspectSentimentModel class
aspect_model = AspectSentimentModel(model_dir="saved_sentiment_model")

# Analyze a new review
review = "The pasta was absolutely delicious, perfectly cooked with a great blend of spices. However, the service was disappointing; the waiter took too long to attend to us, and we had to wait ages for our drinks. The ambiance was charming and cozy, with beautiful decor and soft lighting."

result = aspect_model.analyze_review(review)
print(result)

