from transformers import logging
logging.set_verbosity_error()  #removed this one message saying Device to CPU all the time
from transformers import pipeline

# Initialize zero-shot classifier

classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")     #facebook/bart-large-mnli was too slow so decided to use this one

def categorize(wish):
    categories = ["Object","Wealth","Experiences","Random","Knowledge/Ability"]  #Wish categories
    classification = classifier(sequences=wish, candidate_labels=categories)
    category_choice = classification["labels"][0]  #Picks category with highest score 
    confidence_score = classification["scores"][0]  #Gives confidence score of category with highest score


    return {"category":category_choice, "confidence":confidence_score}
