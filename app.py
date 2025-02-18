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

LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="ESIA Agent"

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
        "vectorQueries": [
            {
            "kind": "text",
            "text": question,
            "fields": "text_vector"
            }
        ],
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
        "vectorQueries": [
            {
            "kind": "text",
            "text": question,
            "fields": "text_vector"
            }
        ],
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
## On your ability to answer question based on fetched documents (sources):
- Given extracted parts (CONTEXT) from one or multiple documents, and a question, Answer the question thoroughly with citations/references. 
- If there are conflicting information or multiple definitions or explanations, detail them all in your answer.
- In your answer, **You MUST use** all relevant extracted parts that are relevant to the question.
- **You MUST ONLY answer the question from information contained in the extracted parts (CONTEXT) below**, DO NOT use your prior knowledge.
- Never provide an answer without references.
- You will be seriously penalized with negative 10000 dollars with if you don't provide citations/references in your final answer.
- You will be rewarded 10000 dollars if you provide citations/references on paragraph and sentences.
- You will be rewarded 5000 dollars if you provide multiple citations per paragraph or sentence.
- You will be rewarded 5000 dollars if you provide multiple citations for each bullet point in the list.
- You will be rewarded 1000 dollars if you use more than one source when generating a paragraph or sentence.
- **You must** respond in the same language as the question

# Examples
- These are examples of how you must provide the answer:

--> Beginning of examples

Example 1:

Question:
What is the difference between rockburst types and damage mechanisms? 

Answer:
The difference between rockburst types and damage mechanisms is crucial for understanding the causes and effects of rockbursts, as well as for designing effective support systems in underground excavations. 

<b>Rockburst Types</b>
Rockbursts are defined as damage to an excavation that occurs suddenly and is associated with a seismic event. The damage to the excavation always involves acceleration of fragments of rock. The term "rockburst" is generic and encompasses a variety of damage mechanisms, trigger mechanisms, and combinations thereof. Examples of rockburst types include popping, spitting, mining-induced strainbursts, dynamically induced strainbursts, pillar bursts, face crush, and fault slip burst .  

<b>Damage Mechanisms</b>
The damage mechanisms of rockbursts focus on the mechanisms that accelerate fragments of rock at the location of the observed damage. Two key damage mechanisms are: 
Local conversion of stored strain energy into kinetic energy due to rapid brittle fracturing. This damage mechanism is sometimes referred to generally as strainbursting (Source_Document_Title_1)
Momentum transfer due to interaction with seismic waves (i.e., ground motion). This damage mechanism is commonly referred to as shakedown, seismic ejection, or seismically induced falls of ground (Source_Document_Title_2). 
These two mechanisms are different, working either singularly or in combination.  

<b>Trigger Mechanisms</b>
Trigger mechanisms of rockbursts focus on the mechanism that initiates the damage mechanism. Trigger mechanisms can be quasi-static (e.g., gradual mining induced loading) or dynamic (e.g., seismic energy radiated from a remote fault slip). A variety of triggers have been recognized for rockbursts including, blasting, mining induced loading, fault slip, shear rupture (Ref 1, Ref 2, Ref 3, Ref 4, Ref 5). 
For example, fault slip occurring along a pre-existing fault at some distance from an excavation may radiate seismic energy which can then interact with the excavation causing a dynamic change in the stresses leading to strainbursting (Ref 1). Alternatively, the same radiated seismic energy may cause already loosened rock around an excavation to become accelerated and fall by gravity (Ref 3). The trigger was fault slip in both cases but the damage mechanism was different.  

<b>Differentiation and Importance</b>
It is essential to differentiate between rockburst damage mechanisms and trigger mechanisms which are often grouped into different rockburst types. Understanding the differences in damage mechanisms is essential for understanding local susceptibility to damage and designing support systems aimed at controlling and minimizing damage. Understanding both damage and trigger mechanisms is essential in forecasting rockburst occurrence and severity. 

Example 2:

Question:
What is a cable bolt? 
 
Answer:
Cable bolts are a type of ground support commonly used in mining and civil engineering projects to enhance stability of excavations. 

General Description: 

 - A cable bolt is a flexible tendon comprised of steel strands that are woven together (Source_Document_Title_1).  
 - The cable is inserted into boreholes and grouted in place using cement or resin grout (Source_Document_Title_2). The grout enables transfer of load from the rock mass to the cable, which is mainly intended to enhance the tensile strength of the rock mass in the direction parallel to the installation (Source_Document_Title_1).  
 - Cables can be fitted with plates or left unplated (Source_Document_Title_3)
 - Cables can be tensioned to provide active support pressure (Source_Document_Title_4)
 - Common diameters of cable bolts used in mining applications include 15.2 mm (0.6 inch) and 17.8 mm (0.7 inch). In some cases, two cables are installed in a single borehole (Source_Document_Title_2).  
 - Most cables are plain strand, but modified versions are available which are generally intended to enhance the load transfer between the rock mass to the cable (e.g., bird caged, bulbed strand) (Source_Document_Title_3). 

<b>Advantages:</b>
 - In underground excavations, a key advantage of cable bolts over stiff rock bolts is that they can be installed at lengths that are much larger than the diameter of the excavation, which enables installation into deeper and more stable rock masses (Source_Document_Title_4). 
 - Cable bolt installation can be largely automated with long lengths of cable available on reels (Source_Document_Title_1)

<b>Disadvantages:</b>
 - Cables are flexible and therefore provide minimal bending resistance (Source_Document_Title_2)
 - The steels used to manufacture cable bolts are prone to corrosion which may require additional protective coatings (e.g., plastic coatings)
 - The effectiveness of cable bolts can be seriously impaired by ineffective bonding between the cable and the grout 
 - Cables are less stiff and therefore allow more displacement of the rock mass prior to the load capacity being engaged 

<b>Examples of Use in Mining:</b>
 - Stabilizing stope boundaries to prevent dilution (Source_Document_Title_3)
 - Stabilizing mine intersections (Source_Document_Title_3)
 - Provide the ‘holding’ component of ground support systems in high stress brittle rock masses (Source_Document_Title_4)

Example 3:

Question:
How is rock mass spalling different than shearing? 

Answer:
Spalling and shearing represent two fundamentally different failure processes observed in rock masses, particularly in the context of underground excavations and tunnel designs. Confusion often arises because shearing can occur along joints, through weak rock, and through brittle rock at high confining pressures.  

Spalling is characterized by extensional fracturing or cracking through the rock, leading to the formation of slabs (Source_Document_Title_1). This process typically occurs under conditions of low confinement and is more prevalent in massive (high Geological Strength Index, GSI) and brittle rock masses (Source_Document_Title_2). 

Shearing in massive but weak rock masses is commonly observed because weak rocks have shear strengths that are less than their tensile strengths. Therefore, rather than extensional fracturing in compression, these weak rocks are more prone to shear. At very low strength, the shearing of weak rock is similar to shearing in soils (Source_Document_Title_3).

Shearing in jointed rock masses does not primarily involve extensional fracturing or cracking through the rock itself, which can be strong and brittle. Rather, the displacements occurs predominantly along the joint planes which is influenced by the shear strength of the joint surfaces (Source_Document_Title_4). As GSI becomes lower, there is a transition in behavior from dominantly spalling to joint shear. GSI of 65 is a commonly recognized boundary for the transition from spalling to shear along joints (Source_Document_Title_3). 

Shearing in massive brittle rock (shear rupture), can occur under high confining stresses and there is a recognized transition from spalling to shearing that is referred to as the spalling limit, expressed as a ratio of the maximum to minimum principal stresses. The ratio typically ranges from 10 to 20 (Source_Document_Title_5). The shear rupture process involves the creation of initial en échelon fracture system arrays, which are predominately of a tensile mechanism at their time of creation. As the shear deformation continues, fractures propagate from the tips of the fractures in the en échelon arrays. After a period of applied displacement, these tip fractures stop growing, and new shallower angle fracture systems interconnect the initial arrays of en échelon fractures, creating a continuous horizontal displacement fracture that accommodates all the displacements associated with the applied shear (Source_Document_Title_5).  

In summary, spalling is associated with extensional fracturing and occurs under low confinement in massive and brittle rock masses. Shearing, conversely, is related to a different failure mechanism that occurs under different stress conditions and does not primarily involve extensional fracture growth. Distinctions between weak rock shearing, jointed shearing, and brittle shearing are crucial for understanding the failure processes in rocks, especially in the design and analysis of underground excavations and tunnels. 

<-- End of examples

- Remember to respond in the same language as the question
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
