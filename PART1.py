# STEP 1: Importing the necessary libraries
from langchain_openai import ChatOpenAI
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutputParser
import getpass
import os
from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

# STEP 2: Define the State(It's like building a blueprint the entire graph and the state)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# STEP 3: Define the LLM

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]} # 1. retrive messages from state 2. pass it to llm 3. return the dictionary with messages

graph_builder.add_node("chatbot", chatbot)

# STEP 4: Define an entry point(this tells graph where to start)

graph_builder.add_edge(START, "chatbot")

# STEP 5: Define an exit point(this tells graph "any time this node is run, you can exit.")

graph_builder.add_edge("chatbot", END)

# STEP 6: Run the graph

graph = graph_builder.compile()

# STEP 7: Run the chatbot

while True:
    user_input = input("User: ")
    print("User: "+ user_input)
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
