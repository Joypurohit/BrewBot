import json
import pandas as pd
import os
from .utils import get_chatbot_response, jsonValidation
from openai import OpenAI
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()


class RecommendationAgent():
    def __init__(self,apriori_recommendation_path,popular_recommendation_path):

        #Load the client with the runpod api key and the deployed chatbot url
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

        #Store the recommendations data from the file that was generated using the apriori algorithm
        with open(apriori_recommendation_path, 'r') as file:
            self.apriori_recommendations = json.load(file)

        #Store the popular recommendations data from the file that was generated using the popularity algorithm
        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        #Read the products and product categories from the popular recommendations dataframe
        self.products = self.popular_recommendations['product'].tolist()
        self.product_categories = self.popular_recommendations['product_category'].tolist()
    
    # Function to get the apriori recommendations based on the products that are provided
    def get_apriori_recommendation(self,products,top_k=5):
        recommendation_list = []

        #Check whichever products is there in the apriori recommendations list, then add the recommendations to the recommendation list
        for product in products:
            if product in self.apriori_recommendations:
                recommendation_list += self.apriori_recommendations[product]
        
        # Sort recommendation list by "confidence" to have the most confident recommendations first
        recommendation_list = sorted(recommendation_list,key=lambda x: x['confidence'],reverse=True)

        recommendations = []
        recommendations_per_category = {}

        # Iterate through the sorted recommendation list and add recommendations to the final list having k recommendations
        for recommendation in recommendation_list:
            # If Duplicated recommendations then skip
            if recommendation in recommendations:
                continue 

            # Fetch the product category and check if it is not in the recommendations_per_category dictionary
            # If it is already in the dictionary and the count is greater than or equal to 2 then skip the recommendation
            product_category = recommendation['product_category']
            if product_category not in recommendations_per_category:
                recommendations_per_category[product_category] = 0
            
            if recommendations_per_category[product_category] >= 2:
                continue

            recommendations_per_category[product_category]+=1

            # Add recommendation
            recommendations.append(recommendation['product'])

            if len(recommendations) >= top_k:
                break

        return recommendations 


    #Function to generate the popular recommendations according to the product categories
    #If the product categories are not provided then it will return the top k popular recommendations
    def get_popular_recommendation(self,product_categories=None,top_k=5):
        recommendations_df = self.popular_recommendations
        
        #If there is only one product category provided then we put it in a list for the further processing part
        if type(product_categories) == str:
            product_categories = [product_categories]

        #Retrieve the product recommendations based on the product categories that are provided and sort them by the number of transactions
        if product_categories is not None:
            recommendations_df = self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        recommendations_df = recommendations_df.sort_values(by='number_of_transactions',ascending=False)
        
        #If there are no recommendations then return an empty list
        if recommendations_df.shape[0] == 0:
            return []

        #Get the top k recommendations from the recommendations dataframe
        recommendations = recommendations_df['product'].tolist()[:top_k]
        return recommendations

    #Function to classify the type of recommendation that is needed based on the user's message
    def recommendation_classification(self,messages):
        system_prompt = """ You are a helpful AI assistant for a coffee shop application which serves drinks and pastries. We have 3 types of recommendations:

        1. Apriori Recommendations: These are recommendations based on the user's order history. We recommend items that are frequently bought together with the items in the user's order.
        2. Popular Recommendations: These are recommendations based on the popularity of items in the coffee shop. We recommend items that are popular among customers.
        3. Popular Recommendations by Category: Here the user asks to recommend them product in a category. Like what coffee do you recommend me to get?. We recommend items that are popular in the category of the user's requested category.
        
        Here is the list of items in the coffee shop:
        """+ ",".join(self.products) + """
        Here is the list of Categories we have in the coffee shop:
        """ + ",".join(self.product_categories) + """

        Your task is to determine which type of recommendation to provide based on the user's message.

        Your output should be in a structured json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
        {
        "chain of thought": Write down your critical thinking about what type of recommendation is this input relevant to.
        "recommendation_type": "apriori" or "popular" or "popular by category". Pick one of those and only write the word.
        "parameters": This is a  python list. It's either a list of of items for apriori recommendations or a list of categories for popular by category recommendations. Leave it empty for popular recommendations. Make sure to use the exact strings from the list of items and categories above.
        }
        """

        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        chatbot_output = jsonValidation(self.client,self.model_name,chatbot_output)
        output = self.postprocess_classfication(chatbot_output)
        return output

    def get_response(self,messages):
        messages = deepcopy(messages)

        #First we classify the type of recommendation that is needed based on the user's message
        recommendation_classification = self.recommendation_classification(messages)
        recommendation_type = recommendation_classification['recommendation_type']
        recommendations = []

        #Based on the recommendation type we get the recommendations from the respective function
        if recommendation_type == "apriori":
            recommendations = self.get_apriori_recommendation(recommendation_classification['parameters'])
        elif recommendation_type == "popular":
            recommendations = self.get_popular_recommendation()
        elif recommendation_type == "popular by category":
            recommendations = self.get_popular_recommendation(recommendation_classification['parameters'])
        
        # If there are no recommendations then return a message saying that we can't help with that
        if recommendations == []:
            return {"role": "assistant", "content":"Sorry, I can't help with that. Can I help you with your order?"}
        
        # Respond to User
        recommendations_str = ", ".join(recommendations)
        
        #Creating a system prompt for the chatbot to recommend items based on the user's order with a small description
        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their input message. And respond in a friendly but concise way. And put it an unordered list with a very small description.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend me those items exactly: {recommendations_str}
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)

        return output


    #Function to postprocess the recommendation classification llm output to have the type and parameters of the recommendation.
    def postprocess_classfication(self,output):
        output = json.loads(output)

        dict_output = {
            "recommendation_type": output['recommendation_type'],
            "parameters": output['parameters'],
        }
        return dict_output

    #To generate recommendations based on whatever the user has ordered
    def get_recommendations_from_order(self,messages,order):
        products = []
        #First we extract the products from the order
        for product in order:
            products.append(product['item'])

        # we apply apriori algorithm on the products in the user order to get the corresponding recommendations
        recommendations = self.get_apriori_recommendation(products)
        recommendations_str = ", ".join(recommendations)

        #And then we design a system prompt to return the recommendations to the user based on their order
        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their order.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend me those items exactly: {recommendations_str}
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self,output):
        output = {
            "role": "assistant",
            "content": output,
            "memory": {"agent":"recommendation_agent"
                      }
        }
        return output

