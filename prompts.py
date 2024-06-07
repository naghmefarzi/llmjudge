

def parse_order(order_str):
    try:
        order = [int(char) for char in order_str]
        if sorted(order) != list(range(len(order))):
            raise ValueError("Invalid order string")
        return order
    except ValueError:
        return "Invalid order string"
    
# Function to create prompts based on different orders
def create_system_message(order_str):
    order = parse_order(order_str)
    descriptions = {
        0: "0 = Irrelevant: The passage has nothing to do with the query.",
        1: "1 = Related: The passage seems related to the query but does not answer it.",
        2: "2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.",
        3: "3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer."
    }
    ordered_descriptions = "\n    ".join(descriptions[i] for i in order)
    prompt = f"""You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale with the following meanings:

    {ordered_descriptions}"""
    print(prompt)
    return prompt
