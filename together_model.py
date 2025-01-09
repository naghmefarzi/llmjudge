# together ai
from together import Together
import together
import os
from typing import *






# prompt = '''Please rate how the given passage is relevant to the query based on the given scores. The output must be only a score that indicates how relevant they are.
# Query: are naturalization records public information
# Passage: From U.S. Citizenship and Immigration Services (USCIS) Naturalization Guide to Naturalization Child Citizenship Act Naturalization Test. Laws and Regulations Read the Code of Federal Regulation Chapter 8 Section 319.2, Expeditious Naturalization regulation and read the INA section 319(b). Department of State Employees and Spouses Only

# Exactness: 0
# Topicality: 1
# Coverage: 1
# Contextual Fit: 1'''
# system_message = '''You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

#     3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
#     2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
#     1 = Related: The passage seems related to the query but does not answer it.
#     0 = Irrelevant: The passage has nothing to do with the query.
#     Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0.'''

# model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"



# together.api_key = os.environ["TOGETHER_API_KEY"]
# client = Together()
# response = client.chat.completions.create(
#     model = model_name,
#     messages = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": prompt},
#     ]    
# )
# print(response.choices[0].message.content)



class TogetherPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is not set.")
        self.client = Together(api_key=self.api_key)
    
    def __call__(self, messages: List[Dict], max_new_tokens=100, **kwargs):
        # Use Together API to generate responses
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return [{"generated_text": response.choices[0].message.content}]
