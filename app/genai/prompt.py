
structured_sentiment_prompt='''
    Instructions:
    Please use the provided text to extract the positive and negative sentiment feedback. 
    The sentiments feedback should encapsulate the most impactful and representative sentiments expressed across the text given.

    Criteria for Sentiment Feedback:

    Relevance: Identify feedback that highlights key strengths, notable features, or exceptional experiences.
    Impact: Prioritize feedback that elicits strong sentiment emotions or signifies significant satisfaction.
    Consolidation: Prefer concise feedback that effectively summarizes broader sentiments.
    Target product: {product}

    Rules:
    1. Strictly extract feedback pointing to {product} only! No other brand or product variant feedback will be included.
        
    {format_instructions}
    Text:{response}
    '''

def get_extract_sentiment_prompt(product, full_transcript):
    prompt = f'''Please perform sentiment analysis on {product}. 
    Only List down the important points which reflects positive sentiment and negative sentiment only. 
    You are given one youtube transcipt. 
    Transcript:{full_transcript}'''
    return prompt

def get_topic_prompt(product, sentiments_list):
    prompt = f'''Please summarize the content in the list within eight words. 
    All the contents are refering to {product}.
    Do not include {product} name in the summary.
    
    List: {sentiments_list}'''
    return prompt
