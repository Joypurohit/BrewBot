def get_chatbot_response(client,messages,model_name,temperature,maxTokens):
    messages_list = []
    for message in messages:
        messages_list.append({"role": message['role'], "content": message['content']})
    response = client.chatCompletion(
        model=model_name,
        messages=messages_list,
        temperature=temperature,
        max_tokens=maxTokens,
        top_p=0.7
    )
    return response.choices[0].message.content

def get_embedding(client,model_name,input_data):
    response = client.embeddings.create(
        model=model_name,
        input=input_data
    )
    embeddings=[]
    for obj in response:
        embeddings.append(obj.embedding)
    return embeddings

#Function to make sure the right json string is generated to be processed by the postProcess function for each agent.
# Each agent generates a json object based on the conditions and is post processed to create a standard final object,
# But this might fail if the llm does not generate a valid json string, so before post processing this jsonValidation function can be used.
def jsonValidation(client,model_name,json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string.

    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_chatbot_response(client,model_name,messages)

    return response