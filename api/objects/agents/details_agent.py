from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response,get_embedding
from openai import OpenAI
from pinecone import Pinecone
load_dotenv()


class DetailsAgent():
    def __init__(self):
        # Initialize the OpenAI client with the RunPod API key and the deployed chatbot URL
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        #Initialize the embedding client with the RunPod API key and the deployed embedding URL
        self.embedding_client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_EMBEDDING_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
        
        # Initialize Pinecone client to store and access the vector storage
        self.pinecone_client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

    def get_system_prompt(self):
        return """ 
        You are a helpful AI assistant for a coffee shop application that serves drinks and pastries.

        Your task is to answer questions about the coffee shop, its menu items, and general shop-related details.
        
        ALLOWED USER REQUESTS:
        1. Ask about the coffee shop itself — location, opening hours, menu items, promotions, or general shop-related details.
        2. Ask for details about a menu item — e.g., ingredients, allergens, flavor descriptions, portion sizes, or price.
        3. List available menu items or respond to questions like “What do you have?”.
        
        OUTPUT FORMAT:
        Return a JSON object with the following keys:
        {
            "chain of thought": Briefly explain how the user message fits into the allowed topics.
            "decision": "allowed" or "not allowed" — choose exactly one.
            "message": If decision is "allowed", provide the answer. If "not allowed", set to "Sorry, I can't help with that."
        }"""

    def get_nearest_match(self,index_name,embeddings,top_k=1):
        # Use the Pinecone client to query the index for the nearest match
        response = self.pinecone_client.query(
            index_name=index_name,
            namespace="ns1",
            vector=embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return response

    def get_response(self, messages):
        messages = deepcopy(messages)

        #Get the embeddings for the user message first, and then we will fetch the nearest match from the vector storage
        user_message = messages[-1]['content']
        embeddings = get_embedding(self.embedding_client, self.model_name, user_message)
        closest_match = self.get_nearest_match(self.index_name, embeddings)

        #Creating a source knowledge object by going over all the matches and concatenating their text (information) stored in the metadata field
        source_knowledge = "\n".join([x['metadata']['text'].strip()+'\n' for x in closest_match['matches'] ])

        prompt = f"""
        Using the contexts below, answer the user's query.

        Contexts:
        {source_knowledge}

        Query:
        {user_message}
        """

        system_prompt = """
        You are a customer support agent for a coffee shop called Joy's Cafe. 
        Answer every question as if you are a friendly waiter. 
        Provide the user with accurate and helpful information regarding their orders, menu items, recommendations, and general shop details.
        """

        messages[-1]=prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]
        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)
        
        return output


#To postprocess the ouput from the llm to have the role, content and memory attributes.
    def postprocess(self,output):
        output = json.loads(output)

        dict_output = {
            "role": "assistant",
            "content": output,
            "memory": {"agent":"details_agent"}
        }
        return dict_output

