def create_qa_prompt(user_input, context_text):
    """
    Create a formatted prompt for BERT QA model

    Args:
        user_input (str): The user's question/input
        context_text (str): The context to search for answers

    Returns:
        str: Formatted prompt ready for BERT QA model
    """
    return f"""
Input: {user_input}
Context: {context_text}
Task: Provide the answer using BERT QA model.
Output:
"""

def demo_prompt_usage():
    """
    Demonstrate how to use the QA prompt template
    """
    # Example context about the whistle service
    context = """
    A whistle is a service that connects local businesses with customers who need their services.
    Users can search for businesses like plumbers, restaurants, salons, mechanics, and tutors.
    Users can also create whistles to notify nearby service providers about specific needs,
    such as tutoring services, rare books, or mechanical repairs.
    The service helps bridge the gap between local service providers and customers in their area.
    """

    # Example user inputs/questions
    user_inputs = [
        "What is a whistle?",
        "What businesses can I search for?",
        "How can I create a whistle?",
        "What services does the platform support?"
    ]

    print("QA Prompt Template Demo")
    print("=" * 50)

    for user_input in user_inputs:
        prompt = create_qa_prompt(user_input, context)
        print(f"Generated Prompt for: '{user_input}'")
        print(prompt)
        print("-" * 50)

    # Show how this would be used with a BERT QA model
    print("\nHow to use with BERT QA model:")
    print("1. Take the generated prompt above")
    print("2. Feed it to a BERT Question Answering model")
    print("3. The model will extract the answer from the context")
    print("\nExample BERT QA usage:")
    print("""
from transformers import pipeline

# Load BERT QA pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Use with context and question
result = qa_pipeline({
    'question': 'What is a whistle?',
    'context': context
})
print(result['answer'])
    """)

if __name__ == "__main__":
    demo_prompt_usage()
