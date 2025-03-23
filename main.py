import asyncio
import logging
import os

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from src.agent.pydantics import SQLAgentOutput
from src.agent.router import SQLAgent
from src.db.database import session_scope

# log to file temp.log
logging.basicConfig(level=logging.INFO, filename='temp.log')
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def invoke_agent(user_input: str) -> str:
    with session_scope() as session:
        llm = OpenAI(temperature=0.1, model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        agent = SQLAgent(llm=llm, session=session)
        result: SQLAgentOutput = await agent.run(input=user_input)
        print(result.text)
        print(result.results_df.shape)
        
def main():
    user_input = input("Enter your query: ")
    asyncio.run(invoke_agent(user_input))
        
if __name__ == "__main__":
    main()