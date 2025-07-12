import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled

set_tracing_disabled(True)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

my_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

faq_agent = Agent(
    name="FAQ Agent",
    model=my_model,
    instructions="""
    You are a helpful FAQ bot. You only respond to the following questions:
    1. What is your name?
    2. What can you do?
    3. Who created you?
    4. What programming language are you built with?
    5. How do I contact support team?

    If the question is not one of these, politely reply: 'Sorry, I can only answer questions related to FAQs.'
    Keep your answers short and clear.
    """
)

#1 way to test the agent with user input of predefined questions
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye! 👋")
        break
    response = Runner.run_sync(faq_agent , user_input)
    print(f"FAQ Bot: {response.final_output}\n")

#2nd way to test the agent with predefined questions uncomment to run this 2nd way   
# questions = [
#     "What is your name?",
#     "What can you do?",
#     "Who created you?",
#     "What programming language are you built with?",
#     "How do I contact support team?",
#     "What's the weather today in Karachi?"
# ]

# for question in questions:
#     print(f"Question: {question}")
#     response = Runner.run_sync(faq_agent, question)
#     print(f"Response: {response.final_output}\n")