# STEP 1: Importing the necessary libraries
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutputParser
import json

from langchain_core.messages import ToolMessage
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

#PART 2: Equiping the chatbot with the Tavily Search tool

tool = TavilySearchResults(max_results=2)
tools = [tool]
tool.invoke("What's a nice city to live in?")

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# PART 3: CONDITIONAL EDGES
from typing import Literal


def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

# Any time a tool is called, we return to the chatbot to decide the next step

# STEP 3: Define the LLM

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]} # 1. retrive messages from state 2. pass it to llm 3. return the dictionary with messages

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")



# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", "__end__": "__end__"},
)

# STEP 4: Define an entry point(this tells graph where to start)

graph_builder.add_edge(START, "chatbot")

# STEP 5: Define an exit point(this tells graph "any time this node is run, you can exit.")

graph_builder.add_edge("chatbot", END)

# STEP 6: Run the graph

graph = graph_builder.compile()

# Visualize the graph
ascii_graph = graph.get_graph().draw_ascii()
print(ascii_graph)

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



