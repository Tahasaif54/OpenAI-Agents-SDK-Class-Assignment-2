import os
import requests
from dotenv import load_dotenv
from agents import (
    Agent, Runner, AsyncOpenAI,
    OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
)

set_tracing_disabled(True)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

if not gemini_api_key or not weather_api_key:
    raise ValueError("Both GEMINI_API_KEY and WEATHER_API_KEY must be set in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

my_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

@function_tool
def add(a: float, b: float):
    """Returns the sum of two numbers."""
    return a + b

@function_tool
def subtract(a: float, b: float):
    """Returns the difference of two numbers."""
    return a - b

@function_tool
def multiply(a: float, b: float):
    """Returns the product of two numbers."""
    return a * b

@function_tool
def divide(a: float, b: float):
    """Returns the quotient of two numbers."""
    if b == 0:
        return "Division by zero is not allowed."
    return a / b

@function_tool
def get_weather(city: str):
    """Fetches real-time weather data for a given city."""
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        location = data["location"]["name"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"The current weather in {location} is {temp_c}°C with {condition.lower()}."
    else:
        return f"Could not retrieve weather info for this {city}."

multi_tool_agent = Agent(
    name="Smart Utility Agent",
    instructions="""
    You are a helpful assistant that can answer math and weather-related questions.
    
    If the user asks something like “What is 5 + 7?” or “Divide 12 by 3”, use the math tools.
    If the user asks something like “What’s the weather in Hyderabad?”, use the weather tool.
    
    Only respond using your tools, and give clear, short answers.
    """,
    model=my_model,
    tools=[add, subtract, multiply, divide, get_weather],
)

prompt = input("Ask your question: ")
result = Runner.run_sync(multi_tool_agent, prompt)
print(f"Response: {result.final_output}")
