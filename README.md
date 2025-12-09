This project simulates a mischievous Genie that responds to user wishes with twists, humor, and wordplay. The Genie aims to learn which types of twists users like the most and improve over time.

cat.py = Wish Categorization module

rl_agent.py = Q-Learning module

response.py = Response Generation module

main.py = Where all the modules are brought together

q_table.json = Q-Table in readable format

How it works:

1.Wish Categorization: When the user enters a wish, the system classifies it into one of the five categories: Object, Wealth, Experience, Random, or Knowledge/Ability. This helps the Genie decide which type of twist to use

2.Selecting a Twist Strategy: Based on the wish category, the Genie picks a "twist strategy" using Q-learning to see which twists are the most humorous. The twist strategies are: Literal, Substitution, Exaggerate, Opposite, and Puns

3.Response Generation: A language model uses the chosen strategy to reply with a funny response, twisting the wish

4.Receiving feedback and improving: The user rates as good, meh, or bad and the genie uses this feedback to update its Q-table and become funnier/more creative

5.Genie's memory: The Genie remembers what it learned by saving its Q-table and learning parameters between sessions to keep improving

Acknowledgements:
- Implemented and debugged primarily by me.
- I used ChatGPT occasionally to help with debugging code and understanding the details of implementation but all core design and logic were developed independently.
- The OpenAI API was used for the response module and it was very cheap.
