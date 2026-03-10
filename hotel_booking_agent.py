from langchain.tools import tool
import json
import redis
import os
from typing import TypedDict, List, Union
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

SESSION_ID = "hotel_user_1"
MAX_TURNS = 5
KEEP_AFTER_SUMMARY = 2
SUMMARY_TRIGGER = 5
TTL_SECONDS = 300


def _turns_key(session_id: str) -> str:
    return f"chat:{session_id}:turns"


def _summary_key(session_id: str) -> str:
    return f"chat:{session_id}:summary"


def get_turns(session_id: str) -> List[dict]:
    raw = redis_client.lrange(_turns_key(session_id), 0, -1)
    return [json.loads(t) for t in raw]


def get_summary(session_id: str) -> str:
    return redis_client.get(_summary_key(session_id)) or ""


def store_turn(session_id: str, turn: dict):
    key = _turns_key(session_id)
    redis_client.rpush(key, json.dumps(turn))
    redis_client.expire(key, TTL_SECONDS)
    redis_client.expire(_summary_key(session_id), TTL_SECONDS)


def update_summary_batch(llm, existing_summary: str, turns_to_summarize: List[dict]) -> str:
    convo_lines = []
    for t in turns_to_summarize:
        convo_lines.append(f"User: {t['user']}")
        convo_lines.append(f"Assistant: {t['assistant']}")
    conversation_text = "\n".join(convo_lines)
    prompt = f"""
You are maintaining a long-term memory summary of a conversation.

CRITICAL RULES:
- MERGE the existing summary with the new conversation turns into ONE unified summary.
- DO NOT append or list summaries separately.
- REMOVE redundancy.
- If new information contradicts old summary, KEEP THE MOST RECENT information.
- Preserve important decisions, preferences, constraints.
- Ignore small talk.
- Keep the result concise and factual.

Existing summary:
{existing_summary}

New conversation turns:
{conversation_text}

Return a single updated summary only.
""".strip()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def build_memory_context(session_id: str) -> List:
    messages = [SystemMessage(content=system_prompt.content)]
    summary = get_summary(session_id)
    if summary:
        messages.append(SystemMessage(content=f"Conversation summary:\n{summary}"))
    for t in get_turns(session_id):
        messages.append(HumanMessage(content=t["user"]))
        messages.append(AIMessage(content=t["assistant"]))
    return messages


def maybe_summarize(session_id: str, llm):
    turns = get_turns(session_id)
    if len(turns) <= SUMMARY_TRIGGER:
        return
    num_to_summarize = len(turns) - KEEP_AFTER_SUMMARY
    turns_to_summarize = turns[:num_to_summarize]
    updated_summary = update_summary_batch(llm, get_summary(session_id), turns_to_summarize)
    redis_client.set(_summary_key(session_id), updated_summary)
    redis_client.ltrim(_turns_key(session_id), num_to_summarize, -1)


def get_cache(key):
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None


def set_cache(key, value, ttl):
    redis_client.setex(key, ttl, json.dumps(value))


SEARCH_DATA = {
    "city": "Jaipur",
    "hotels": [
        {"hotel_id": "HTL001", "name": "Maharani Palace", "stars": 5, "location": "MI Road", "base_price": 4500},
        {"hotel_id": "HTL002", "name": "Royal Haveli", "stars": 4, "location": "Pink City", "base_price": 2800},
        {"hotel_id": "HTL003", "name": "Budget Stay", "stars": 2, "location": "Sindhi Camp", "base_price": 900}
    ]
}

AVAILABILITY_DATA = {
    "hotel_id": "HTL001",
    "hotel_name": "Maharani Palace",
    "pricingByRoom": [
        {
            "roomType": "Deluxe Room",
            "maxAvailableRoom": 3,
            "data": [
                {"category": "Room Only", "totalPrice": 4725, "basePrice": 4500},
                {"category": "Room With Breakfast", "totalPrice": 5512, "basePrice": 5250},
                {"category": "Room With All Meals", "totalPrice": 6562, "basePrice": 6250}
            ]
        },
        {
            "roomType": "Suite",
            "maxAvailableRoom": 1,
            "data": [
                {"category": "Room Only", "totalPrice": 8925, "basePrice": 8500},
                {"category": "Room With Breakfast", "totalPrice": 10237, "basePrice": 9750}
            ]
        }
    ]
}

HOTEL_DETAILS = {
    "hotel_id": "HTL001",
    "name": "Maharani Palace",
    "address": "MI Road, Jaipur, Rajasthan 302001",
    "amenities": ["Pool", "Spa", "Free WiFi", "Restaurant", "Valet Parking"],
    "policies": {
        "checkin": "14:00",
        "checkout": "11:00",
        "cancellation": "Free cancellation up to 24 hours before check-in",
        "pets": "Not allowed"
    },
    "nearby": ["Hawa Mahal – 2km", "Amber Fort – 11km", "Jaipur Junction – 3km"]
}


@tool
def search_hotels(city: str, checkin: str, checkout: str, guests: int) -> str:
    """
    Search hotels by city and dates.

    Args:
        city: Name of the city to search hotels in.
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.
        guests: Number of guests.

    Returns:
        List of available hotels in that city with pricing.
    """
    cache_key = f"hotel:search:{city}:{checkin}:{checkout}"
    cached = get_cache(cache_key)
    if cached:
        return json.dumps({"source": "cache", "data": cached})
    result = SEARCH_DATA
    set_cache(cache_key, result, 600)
    return json.dumps({"source": "api", "data": result})


@tool
def check_availability(hotel_id: str, checkin: str, checkout: str) -> str:
    """
    Check room availability and pricing for a specific hotel.

    Args:
        hotel_id: Unique ID of the hotel (e.g. HTL001).
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.

    Returns:
        Room types, availability count, and pricing breakdown.
    """
    cache_key = f"hotel:availability:{hotel_id}:{checkin}:{checkout}"
    cached = get_cache(cache_key)
    if cached:
        return json.dumps({"source": "cache", "data": cached})
    result = AVAILABILITY_DATA
    set_cache(cache_key, result, 300)
    return json.dumps({"source": "api", "data": result})


@tool
def get_hotel_details(hotel_id: str) -> str:
    """
    Get amenities, policies, and location details for a specific hotel.

    Args:
        hotel_id: Unique ID of the hotel (e.g. HTL001).

    Returns:
        Hotel amenities, check-in/check-out policies, cancellation policy, and nearby landmarks.
    """
    cache_key = f"hotel:details:{hotel_id}"
    cached = get_cache(cache_key)
    if cached:
        return json.dumps({"source": "cache", "data": cached})
    result = HOTEL_DETAILS
    set_cache(cache_key, result, 1800)
    return json.dumps({"source": "api", "data": result})


tools = [search_hotels, check_availability, get_hotel_details]


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)


def model_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


def router(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "call_tool"
    return END


def tool_node_wrapper(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    new_messages = list(state["messages"])
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        print(f"Tool used: {tool_name}")
        print(f"Tool input: {tool_input}")
        tool_func = next((t for t in tools if t.name == tool_name), None)
        if tool_func:
            result = tool_func.invoke(tool_input)
        else:
            result = f"Tool {tool_name} not found"
        new_messages.append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": new_messages}


workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.add_node("call_tool", tool_node_wrapper)
workflow.set_entry_point("model")
workflow.add_conditional_edges("model", router)
workflow.add_edge("call_tool", "model")
app = workflow.compile()


system_prompt = SystemMessage(content="""
You are a Hotel Booking AI Agent.

Your job is to help users search hotels, check availability, and get hotel details.

You have access to three tools.

---

1️⃣ search_hotels

Use this tool when the user wants to find hotels in a city.

Input:
- city
- checkin
- checkout
- guests

Example:
User: Find hotels in Jaipur for 2 guests from Oct 10 to Oct 12.

---

2️⃣ check_availability

Use this tool when the user asks about room availability or pricing for a specific hotel.

Input:
- hotel_id
- checkin
- checkout

Example:
User: Check availability for Maharani Palace.

---

3️⃣ get_hotel_details

Use this tool when the user asks about amenities, policies, or location.

Input:
- hotel_id

Example:
User: Tell me more about Maharani Palace.

---

Guidelines:

• Always choose the correct tool when required.
• Extract the correct parameters from the user query.
• If the user already has hotel_id, use it.
• If user asks general questions without tool usage, answer normally.

Examples:

User: Find hotels in Jaipur from Oct 10 to Oct 12 for 2 guests  
→ call search_hotels

User: Show availability for HTL001  
→ call check_availability

User: What amenities does HTL001 have?  
→ call get_hotel_details

Be concise and helpful.
""")


def interactive_conversation():
    print("🏨 Hotel Booking AI Agent Started")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = build_memory_context(SESSION_ID)
        messages.append(HumanMessage(content=user_input))

        state = {"messages": messages}
        final_state = app.invoke(state)

        ai_response = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                print(f"\nAI: {msg.content}\n")
                break

        store_turn(SESSION_ID, {"user": user_input, "assistant": ai_response})
        maybe_summarize(SESSION_ID, llm)


if __name__ == "__main__":
    interactive_conversation()
