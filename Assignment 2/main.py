import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool , set_tracing_disabled

set_tracing_disabled(True)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

my_model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

@function_tool
def add(a,b):
    """This Function Adds Two Numbers."""
    return a+b

@function_tool
def subtract(a,b):
    """This Function Subtracts Two Numbers."""
    return a-b

@function_tool
def multiply(a,b):
    """This Function Multiplies Two Numbers."""
    return a*b

@function_tool
def divide(a,b):
    """This Function Divides Two Numbers."""
    return a / b

math_agent = Agent(
    name="Math Tool Agent",
    instructions="""
    You are a smart math assistant.
    Use your tools to perform calculations like addition substraction multiplication and division when asked.
    Only use tools when necessary.
    """,
    tools=[add,subtract,multiply,divide],
    model=my_model
)
#1st way
prompt = input("Enter a math question: ")
result = Runner.run_sync(math_agent, prompt)
print(f"Response: {result.final_output}")

#2nd way
# questions = [
#     "What is 5 + 7?",
#     "Can you multiply 9 by 6?",
#     "What’s the result of 15 plus 4?",
#     "Tell me the product of 10 and 3."
# ]

# for question in questions:
#     print(f"User: {question}")
#     result = Runner.run_sync(math_agent, question)
#     print(f"Response: {result.final_output}\n")