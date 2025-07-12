import os
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool , set_tracing_disabled

set_tracing_disabled(True)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

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
def get_weather(city: str):
    """
    Get the current weather for a given city using weatherapi.com.
    """
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={city}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        location = data["location"]["name"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"The current weather in {location} is {temp_c}°C with {condition.lower()}."
    else:
        return f"Could not fetch weather data fo this {city}."

weather_agent = Agent(
    name = "Weather Agent",
    instructions="""
    You are a weather agent. If a user asks about the weather in any city, 
    use the get_weather tool to provide the current weather information for that city.
    If the user asks anything else not related to weather, reply: 
    'I am a weather agent, I can only provide weather information.'
    """,
    model=my_model,
    tools=[get_weather],
)

prompt = input("Enter your question: ")
result = Runner.run_sync(weather_agent, prompt)
print(f"Response: {result.final_output}")