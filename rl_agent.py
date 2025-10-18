import numpy as np
import random
#Insert possible easter eggs here and there?
import pickle

def save_q_table(filename="q_table.pkl"):
    global qTable
    with open(filename, "wb") as f:
        pickle.dump(qTable, f)
    print("Q-table saved!")

def load_q_table(filename="q_table.pkl"):
    global qTablej
    try:
        with open(filename, "rb") as f:
            qTable = pickle.load(f)
        print("Q-table loaded!")
    except FileNotFoundError:
        print("No saved Q-table found. Using default Q-table.")

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
    return epsilon

def decay_epsilon():
    global epsilon
    epsilon = max(minEpsilon, epsilon * epsilonDecay)
    return epsilon

#State labels
stateLabels = ["Object","Wealth","Experiences","Random","Knowledge/Ability"]

#Action labels
actionLabels = ["Literal","Substitution","Exaggerate","Opposite","Puns"]

#Initialize the Q table
qTable = {}
for state in stateLabels:
    qTable[state] = {}
    for action in actionLabels:
        qTable[state][action] = 0

#Choose action based on the state!
def chooseAction(state, qTable, epsilon):
    if random.random() < epsilon:
        chosenAction = random.choice(list(qTable[state].keys()))  #Each state is basically a dictionary of it's own with the keys as the twist strategies
        #and the values as the number of points accumulated for those strategies. So .keys() brings out the strategies and we turn it into a list to select one of them randomly
    else:
        maxValue = max(qTable[state].values())   #Lists all values from each strategy and finds the maximum one
        stateActions = qTable[state].items()  # returns list of (action, Q-value) pairs for that particular wish category

        bestActions = []  #To store the best actions

        for action, value in stateActions:
            if value == maxValue:          #Collect all the highest Q-value strategies and add them to bestActions
                bestActions.append(action)  

        chosenAction = random.choice(bestActions)# Randomly pick one if there are multiple top strategies
    
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


