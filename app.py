from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import AgentExecutor, create_react_agent, load_tools
import os
from langgraph.types import Command
import requests
from typing import Literal
from langchain_openai import AzureChatOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv("azure_credentials.env")

from langgraph.graph import MessagesState, END

class State(MessagesState):
    next: str 
    
azure_endpoint = st.secrets["azure_endpoint"]
openai_api_key = st.secrets["openai_api_key"]
openai_deployment_name = st.secrets["openai_deployment_name"]
openai_api_version =st.secrets["openai_api_version"]

llm = AzureChatOpenAI(
    deployment_name=openai_deployment_name, 
    openai_api_version=openai_api_version, 
    openai_api_key=openai_api_key, 
    azure_endpoint=azure_endpoint, 
    temperature=0
)

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

service_endpoint = st.secrets["service_endpoint"]
index_name = st.secrets["index_name"]
key = st.secrets["key"]
search_service_name = st.secrets["search_service_name"]
search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))

def esia_split_results(question) -> str:
    """Searches ESIA documents in a vector store."""
    
    url = f"https://esg-semantic-ranker.search.windows.net/indexes/guinea-esia-2/docs/search?api-version=2024-11-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }

    response = requests.post(url, headers=headers, verify=False, json={
        "search": question,
        "semanticConfiguration":"guinea-esia-2-semantic-configuration",
        "queryType":"semantic",
        "queryLanguage":"en-US",
        "select": "title,chunk",
        "top": 5
    })
    
    results_list = response.json()['value']
    
    return results_list

@tool
def esia_search(question) -> str:
    """Searches ESIA documents in a vector store."""
    
    url = f"https://esg-semantic-ranker.search.windows.net/indexes/guinea-esia-2/docs/search?api-version=2024-11-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }

    response = requests.post(url, headers=headers, verify=False, json={
        "search": question,
        "semanticConfiguration":"guinea-esia-2-semantic-configuration",
        "queryType":"semantic",
        "queryLanguage":"en-US",
        "select": "title,chunk",
        "top": 5
    })

    print(response)
    
    results = response.json()['value']

    print(results)
    
    return str(results)

instructions = """
You are an assistant that searches for information in a database of ESIA documents.
Use the esia_search tool to get information from the database.

### INSTRUCTIONS ### 
When you generate a response, please make sure to provide a reference to the source of the information. 

### EXAMPLE ###
The Simandou project is a major iron ore mining initiative located in the Simandou mountain range in Guinea (1). 
It involves the extraction of iron ore from the Ouéléba deposit, the construction of a 73 km railway to connect the mine to the Trans-Guinea railway, and the development of port facilities on the Morébayah River for exporting the mined ore (2). 
The project is being developed by Rio Tinto Simfer and the Winning Consortium Simandou (WCS), which are responsible for different components of the infrastructure, including rail and port facilities.

References:
1. Dredging and Spoil Disposal Management Plan_Rev0_FR.pdf
2. Coastal Management Plan_Rev 1_Marine.pdf
"""

from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    llm, tools=[esia_search], prompt=instructions
)

def research_node(state: State) -> Command[Literal["__end__"]]:
    
    result = research_agent.invoke(state)
        
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto=END,
    )
    
builder = StateGraph(State)
builder.add_edge(START, "researcher")
builder.add_node("researcher", research_node)

graph = builder.compile()

st.set_page_config(page_title="Guinea ESIA Assistant", page_icon="🔎")
st.title("🔎 Guinea ESIA Assistant")

# DIsplay the search results in the sidebar
st.sidebar.write("## 🏅 SEARCH RESULTS")

if user_question := st.chat_input(placeholder="What is the Simandou project?"):
    st.chat_message("user").write(user_question)
    with st.chat_message("assistant"):
#       st_callback = StreamlitCallbackHandler(st.container())
        with st.spinner("Searching for information..."):
            cycle = graph.stream(
                {
                    "messages": [
                        (
                            "user",
                            user_question,
                        )
                    ]
                },
                #subgraphs=True,
                stream_mode=["updates"]   
            )
            
            for s in cycle:
                print()
                
            st.write(s[-1]["researcher"]["messages"][-1].content)
            
            res_list = esia_split_results(user_question)
            
            i=0
            for result in res_list:
                i+=1
                with st.sidebar.expander("Retrieved result #" + str(i)):
                    st.write("## Document Name:")
                    st.write(result['title'])
                    st.write("## Document Chunk:")
                    st.write(result['chunk'])
