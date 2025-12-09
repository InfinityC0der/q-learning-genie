import numpy as np
import random
#Insert possible easter eggs here and there?
import pickle
import json
import pandas as pd

def get_q_table():  # Just to get the q table for main.py
    global qTable
    return qTable

def save_q_table(pickle_file="q_table.pkl", json_file="q_table.json"):
    global qTable
    #first save as pickle(speed)
    with open(pickle_file, "wb") as f:
        pickle.dump(qTable, f)
    print("Q-table saved to pickle!")
    #Now save to JSON(for readability)
    with open(json_file, "w") as f:
            json.dump(qTable, f, indent=2)
    print(f"Q-table also saved as JSON to {json_file}!")
    #Now let's save the Q-table
    df = pd.DataFrame(qTable).T  #transpose states into rows
    print("\n=== Current Q-table ===")
    print(df)
    print("=====================")

def load_q_table(pickle_file="q_table.pkl", json_file="q_table.json"):
    global qTable
    try:
        #load pickle if it exists
        with open(pickle_file, "rb") as f:
            qTable = pickle.load(f)
        print("Q-table loaded from {pickle_file}!")
    except FileNotFoundError:
        print("No saved Q-table found in {pickle_file}.")
        qTable = initializeQ()

    #Try loading JSON if it exists
    try:
        with open(json_file, "r") as f:
            json_table = json.load(f)
        print(f"Q-table JSON loaded from {json_file} (readable/editable.)")
        #Print table after loading:
        df = pd.DataFrame(qTable).T
        print(df)
        print("======================\n")
    except:
        print(f"No JSON Q-table found at {json_file}.")


#Hyperparameters
epsilon = 1
epsilonDecay = 0.98
minEpsilon = 0.01
alpha = 0.8 
gamma = 0.96


def save_epsilon(filename="epsilon.pkl"):
    global epsilon
    with open(filename, "wb") as f:
        pickle.dump(epsilon, f)

def load_epsilon(filename="epsilon.pkl"):
    global epsilon
    try:
        with open(filename, "rb") as f:
            epsilon = pickle.load(f)
    except FileNotFoundError:
        epsilon = 1

def get_epsilon():
    global epsilon
    print(f"epsilon value being used is {epsilon}")
    return epsilon

def decay_epsilon():
    global epsilon
    epsilon = max(minEpsilon, epsilon * epsilonDecay)
    print(f"epsilon value has been updated:{epsilon}")
    return epsilon

#State labels
stateLabels = ["Physical Item","Money","Powers/Abilities","Random"]

#Action labels
actionLabels = ["Literal","Substitution","Exaggerate","Opposite","Puns"]

#Q table initialization function
def initializeQ():
    qTable = {}
    for state in stateLabels:
        qTable[state] = {}
        for action in actionLabels:
            qTable[state][action] = 0
    return qTable

#Choose action based on the state!
def chooseAction(state, qTable, epsilon):
    if random.random() < epsilon:
        chosenAction = random.choice(list(qTable[state].keys()))  #Each state is basically a dictionary of it's own with the keys as the twist strategies
        #and the values as the number of points accumulated for those strategies. So .keys() brings out the strategies and we turn it into a list to select one of them randomly
        print(f"Exploration: Chose random twist strategy '{chosenAction}' for state '{state}'")  #Here I put the print statement just to verify what the genie is doing.
    else:
        maxValue = max(qTable[state].values())   #Lists all values from each strategy and finds the maximum one
        stateActions = qTable[state].items()  # returns list of (action, Q-value) pairs for that particular wish category

        bestActions = []  #To store the best actions

        for action, value in stateActions:
            if value == maxValue:          #Collect all the highest Q-value strategies and add them to bestActions
                bestActions.append(action)  

        chosenAction = random.choice(bestActions)# Randomly pick one if there are multiple top strategies
        print(f"Exploitation: Chose best twist strategy '{chosenAction}' for state '{state}' with Q-value {maxValue}")  #Here the genie decides to go with exploitation and choose the best strategy
    return chosenAction #return the actual action!

#Get feedback
def get_feedback():

    while True:
        userInput = input("Rate the Genie's response (g = good, m = meh, b = bad): ").lower()
        if userInput in ['g', 'm', 'b']:
            break
        print("Invalid input. Please enter 'g', 'm', or 'b'.")

    # Map input to reward
    feedback = {'g': 1, 'm': 0, 'b': -1}
    reward = feedback[userInput]
    return reward

#Use Bellman's Equation to update Q value
def update_q_table(state, action, reward):
    # For your Genie, next state can just be current state
    next_state = state
    max_future_q = max(qTable[next_state].values())
    
    # Bellman update
    qTable[state][action] = qTable[state][action] + alpha * (reward + gamma * max_future_q - qTable[state][action])




