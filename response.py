import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()#Loading the .env file to get the API key

api_key = os.getenv("key")
client = OpenAI(api_key=api_key)#Here we use the OpenAI constructor because the client object NEEDS the key and we set the parameter equal to the variable

ChatModel = "gpt-4.1-mini"

#Twist Strategies
actionToPrompts = {
    "Literal":"Take the wish literally and make it backfire humorously.",
    "Substitution":"Substitute the wish with something similar that backfires humorously",
    "Exaggerate":"Exaggerate the wish to make it backfire humorously",
    "Opposite": "Grant the opposite of the wish to make it backfire humorously",
    "Puns": "Make the wish backfire with wordplay or puns"
}

def response(wish, action):
    twistAction = actionToPrompts[action]

    prompt = f"""
    You're a mischievous genie.
    {twistAction}
Wish: "{wish}"
Respond in 1-2 hilarious sentences.
""" 
    # Call Chat Completions API
    res = client.chat.completions.create(
        model=ChatModel,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=100
    )

    # Extract the text from the response
    return res.choices[0].message.content.strip()

def main():
    test_wish = "I want to fly"
    test_action = "Exaggerate"

    result = response(test_wish, test_action)
    print("Genie response:", result)

#main()



