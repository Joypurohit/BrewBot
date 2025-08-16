
from agents import GuardAgent, ClassificationAgent,DetailsAgent,AgentProtocol,RecommendationAgent,OrderTakingAgent
import os
from typing import Dict

def main():
    pass

if __name__ == "__main__":
    guardAgent = GuardAgent()
    classificationAgent = ClassificationAgent()
    recommendationAgent = RecommendationAgent("api\objects\recommendation_data\apriori_recommendations.json","api\objects\recommendation_data\popular_recommendations.csv")

    #To enforce the agent protocol standards for the computing agents
    agents_dict : Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": recommendationAgent,
        "order_taking_agent": OrderTakingAgent(recommendation_agent=recommendationAgent)
    }


    messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        #Get user input
        user_prompt =input("User: ")
        messages.append({"role": "user", "content": user_prompt})

        #Get response from the guard agent
        guardAgent_response = guardAgent.get_response(messages)
        print("Guard Agent Response: ", guardAgent_response)

        #We only populate the messages when the user query is not allowed
        if guardAgent_response['memory']['guard_decision'] == "not allowed":
            # Append the response from the guard agent to the messages
            messages.append(guardAgent_response)

        #Get response from the classification agent
        classificationAgent_response = classificationAgent.get_response(messages)
        classified_agent = classificationAgent_response['memory']['classification_decision']
        print("Executing {} agent".format(classified_agent))

        #Get the response from the classified agent
        agent_response = agents_dict[classified_agent].get_response(messages)
        messages.append(agent_response)



if __name__ == "__main__":
    main()

