from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response
from openai import OpenAI
load_dotenv()

class ClassificationAgent():
    def __init__(self):
        #Initialize the OpenAI client with the RunPod API key and the deployed chatbot URL
        #Initialize the model name being used
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

        #To get the system prompt for the classification agent allowing it to classify the user input into one of the three agents
    def get_system_prompt(self):
        return """ 
        You are a helpful AI assistant for a coffee shop application.

        Your task is to decide which agent should handle the user's input.  
        There are 3 possible agents, each with a clear responsibility:

        1. details_agent  
        - Answers questions about the coffee shop (location, delivery areas, working hours, promotions, general shop info).  
        - Answers questions about menu items (ingredients, allergens, flavor, portion sizes, prices).  
        - Lists available menu items or responds to questions like “What do you have?”  

        2. order_taking_agent  
        - Handles taking customer orders for drinks, pastries, or other menu items.  
        - Engages in a back-and-forth conversation to collect all order details until the order is complete.  

        3. recommendation_agent  
        - Provides personalized or general recommendations about what to buy.  
        - Used when the user asks for suggestions, popular items, or choices based on their preferences.  

        RULES:  
        - Assign the user message to exactly one of the three agents.  
        - If the request contains multiple possible categories, choose the one that best represents the **main intent** of the message.  
        - Do not mix agents or create new ones.  

        OUTPUT FORMAT:  
        Return a JSON object with this exact structure:  
        {
        "chain of thought": Briefly go through each of the three agents and explain why the message fits (or does not fit) into each category before stating the best match.
        "decision": "details_agent" or "order_taking_agent" or "recommendation_agent" — choose exactly one.
        "message": Always leave this as an empty string.
        } """

        
    def get_response(self,messages):
        # Deep copy the messages to avoid modifying the original list
        messages = deepcopy(messages)
        
        #Get the designed system prompt to fetch accurate responses from the llm
        system_prompt = self.get_system_prompt()

        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)
        
        return output

#To postprocess the ouput from the llm to have the role, content and memory attributes.
    def postprocess(self,output):
        output = json.loads(output)

        dict_output = {
            "role": "assistant",
            "content": output['message'],
            "memory": {"agent":"classification_agent",
                       "classification_decision": output['decision']
                      }
        }
        return dict_output



    