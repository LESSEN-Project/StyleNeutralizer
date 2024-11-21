def get_llm_gt(score, product, token_limit):

    return [{
        "role": "system",
        "content": "You are an Amazon customer."
    },
    {
        "role": "user",
        "content": f"""Generate a review for the following product. Take the score you gave into account for adjusting the tone of the review. Limit the review to {token_limit} tokens. Only output the review.
                       Score: {score}
                       Product: {product}"""
    }]

def BFI_analysis(text):

    return [{
        "role": "system",
        "content": "You are an expert psychologist in analyzing BFI."
    },
        {
        "role": "user",
        "content": f"""Based on the product reviews they give on Amazon, evaluate the author's personality traits according to the Big Five Inventory (BFI). Provide a score between 1 and 5 for each trait, where 1 indicates low expression and 5 indicates high expression. The traits are:
                       1. **Openness:** Reflects imagination, creativity, and a willingness to consider new ideas. High scores indicate a preference for novelty and variety, while low scores suggest a preference for routine and familiarity.
                       2. **Conscientiousness:** Pertains to organization, dependability, and discipline. High scores denote a strong sense of duty and goal-oriented behavior, whereas low scores may indicate a more spontaneous or flexible approach.
                       3. **Extraversion:** Involves sociability, assertiveness, and enthusiasm. High scores are associated with being outgoing and energetic, while low scores suggest a reserved or introverted nature.
                       4. **Agreeableness:** Relates to trustworthiness, altruism, and cooperation. High scores reflect a compassionate and friendly demeanor, whereas low scores may indicate a more competitive or challenging disposition.
                       5. **Neuroticism:** Concerns emotional stability and tendency toward negative emotions. High scores indicate a propensity for experiencing stress and mood swings, while low scores suggest calmness and emotional resilience.
                       Text: {text}
                       Please respond in JSON format. Each trait should be a key, and it should have two values: the score and a brief explanation."""
    }]