from rl_agent import chooseAction, update_q_table, get_feedback, qTable, load_q_table, save_q_table
from rl_agent import load_epsilon, save_epsilon, get_epsilon, decay_epsilon
from cat import categorize
import response
epsilon = 1
load_q_table()
load_epsilon()
def main():
    print("Hello user, I am a genie trapped in this computer and I'll grant you one wish") #Perhaps add an ASCll genie later for images
    print("But as they say, be careful what you wish for!")
    wish1 = input("Make your 1st wish:\n")

    
    stateInfo = categorize(wish1)
    categoryName = stateInfo["category"]  #wish category is also the states
    confidence = stateInfo["confidence"]
    print(categoryName, confidence)            #The model is so insanely slow that I may need to put all three wishes together separated by a hash symbol or something
    epsilon = get_epsilon()
    action = chooseAction(categoryName, qTable, epsilon) #Choose twist strategy based on the wish category
    response1 = response.response(wish1, action)      #Generate response
    print("Genie: For your 1st wish:", response1)   #Print response!
    reward = get_feedback()   #Get da feedback
    update_q_table(categoryName, action, reward) #Update the qTable with Bellman EQUATION
    decay_epsilon()  #Decay epsilon after updating qTable
    save_q_table()
    save_epsilon()
if __name__ == "__main__":
    main()