from transformers import logging
logging.set_verbosity_error()  #removed this one message saying Device to CPU all the time
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="typeform/mobilebert-uncased-mnli")     #facebook/bart-large-mnli was too slow so decided to use this one

def categorize(wish):
    categories = ["Physical Item","Money","Powers/Abilities","Random"]  #Wish categories
    classification = classifier(sequences=wish, candidate_labels=categories)
    category_choice = classification["labels"][0]  #Picks category with highest score 
    confidence_score = classification["scores"][0]
    print(f"Predicted Category: {category_choice}")
    print(f"Confidence Score: {confidence_score}")
    return {"category":category_choice, "confidence":confidence_score}

#Just a quick little test script
def main():
    test_wish = "I wish I had the ability to teleport"
    categorize(test_wish)

#main()

