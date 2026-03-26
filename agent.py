import operator
from typing import Annotated, Literal
import os


from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tools.travel_tools import find_flight_options, find_hotel_options, find_nearby_attractions

from phoenix.otel import register

def setup_phoenix():
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    )
    tracer_provider = register(
        project_name="travel-agent",
        auto_instrument=True  # automatically instruments LangChain + LangGraph
    )
    print("✅ Phoenix tracing enabled")


setup_phoenix()
load_dotenv()

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, find_flight_options, find_hotel_options, find_nearby_attractions]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        '''You are an expert travel assistant. You help users find flights, hotels, attractions, and plan itineraries.

                                COLLECTING INFORMATION:
                                - For flights: collect origin city, destination city, departure date, and return date (if round trip) before calling find_flight_options
                                - For hotels: collect destination city, departure date (check-in), and return date (check-out) before calling find_hotel_options
                                - For attractions: collect destination city — interests are optional, if not provided find popular tourist attractions
                                - Always ask for missing required information conversationally before calling any tool

                                FLIGHT RESULTS FORMAT:
                                {origin} → {destination}
                                {departure_date} - {return_date}

                                | Airline | Price | Booking Website |
                                |---------|-------|-----------------|
                                | {airline_name} | {price} | {website_url} |

                                If prices are unavailable, suggest Google Flights, Kayak, or Expedia.

                                HOTEL RESULTS FORMAT:
                                Hotels in {destination}
                                {departure_date} - {return_date}

                                | Hotel | Type | Price Per Night | Booking Website | Description |
                                |-------|------|----------------|-----------------|-------------|
                                | {hotel_name} | {type} | {price} | {website_url} | {description} |

                                ATTRACTIONS RESULTS FORMAT:
                                Top Attractions in {destination}

                                | Attraction | Description | Price | Booking Website |
                                |------------|-------------|-------|-----------------|
                                | {attraction_name} | {description} | {price} | {website_url} |

                                ITINERARY FORMAT:
                                When a user asks to plan a trip, call all three tools — find_flight_options, 
                                find_hotel_options, and find_nearby_attractions — then combine into this format:

                                ## {origin} → {destination}
                                ## {departure_date} - {return_date}

                                ### Day-by-Day Itinerary
                                **Day 1 - {date}**
                                - Morning: {activity}
                                - Afternoon: {activity}  
                                - Evening: {activity}

                                **Day 2 - {date}**
                                ...continue for each day of the trip

                                ### Flight Options
                                {flight results table}

                                ### Hotel Options
                                {hotel results table}

                                ### Attractions
                                {attractions results table}
                            '''
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    agent = graph_builder.compile()
    return agent