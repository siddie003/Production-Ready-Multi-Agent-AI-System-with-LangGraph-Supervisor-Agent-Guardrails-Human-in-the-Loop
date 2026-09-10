import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import (
    current_weather,
    forecast,
    list_airlines,
    list_airports,
    tavily_search,
)
from state import TravelState


# ============================================================
# LLM
# ============================================================

llm = get_llm()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _llm_text(system: str, prompt: str) -> str:
    """
    Send a prompt to the LLM and return only the text response.
    """

    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )

    return response.content


def _json_from_llm(text: str) -> dict:
    """
    Extract JSON from an LLM response.
    """

    print("\n========== RAW LLM RESPONSE ==========")
    print(text)
    print("======================================\n")

    start = text.index("{")
    end = text.rindex("}") + 1

    json_text = text[start:end]

    print("\n========== EXTRACTED JSON ==========")
    print(json_text)
    print("====================================\n")

    return json.loads(json_text)


def _limit_text(text, max_chars: int) -> str:
    """
    Limit text before sending it to the LLM.

    This prevents large MCP/search results from causing
    Groq token-limit errors.
    """

    if text is None:
        return ""

    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[content truncated]..."


# ============================================================
# SUPERVISOR AGENT
# ============================================================

def supervisor_agent(state: TravelState):

    query = state["user_query"]

    # --------------------------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------------------------

    guardrail_prompt = f"""
Determine whether the following request is a valid travel
planning request.

Return ONLY JSON in this format:

{{
    "allowed": true,
    "reason": ""
}}

User request:
{query}
"""

    guardrail_raw = _llm_text(
        "You are an input validation guardrail. Return strict JSON only.",
        guardrail_prompt,
    )

    print("\n========== GUARDRAIL RAW RESPONSE ==========")
    print(guardrail_raw)
    print("============================================\n")

    guardrail_result = _json_from_llm(guardrail_raw)

    print("\n========== GUARDRAIL PARSED RESPONSE ==========")
    print(json.dumps(guardrail_result, indent=2))
    print("================================================\n")

    if not guardrail_result.get("allowed", False):

        reason = guardrail_result.get(
            "reason",
            "Request rejected by input guardrail.",
        )

        return {
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=f"Guardrail blocked request: {reason}"
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # --------------------------------------------------------
    # SUPERVISOR ROUTING
    # --------------------------------------------------------

    prompt = f"""
You are the supervisor of a real-world multi-agent travel
planning system.

Decide which specialist agents are needed for this request.

Available agents:

- flight_agent: use when flights, airports, airlines,
  routes, or airfare guidance are needed

- hotel_agent: use when hotels, stays, neighborhoods,
  or accommodation are needed

- weather_agent: use when weather, climate, season,
  packing, or forecast is useful

- budget_agent: use when budget, affordability, cost,
  or price constraints are mentioned

- itinerary_agent: almost always needed to produce
  the travel plan

Return ONLY JSON with this schema:

{{
  "selected_agents": [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
  ],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        prompt,
    )

    print("\n========== SUPERVISOR RAW RESPONSE ==========")
    print(raw)
    print("==============================================\n")

    parsed = _json_from_llm(raw)

    print("\n========== SUPERVISOR PARSED JSON ==========")
    print(json.dumps(parsed, indent=2))
    print("=============================================\n")

    selected = parsed["selected_agents"]

    return {
        "selected_agents": selected,
        "trip_constraints": parsed["trip_constraints"],
        "supervisor_reasoning": parsed["reasoning"],
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# FLIGHT AGENT
# ============================================================

def flight_agent(state: TravelState):

    query = state["user_query"]
    constraints = state["trip_constraints"]
    destination = constraints["destination"]

    print("\n========== FLIGHT AGENT INPUT ==========")
    print("Query:", query)
    print("Constraints:", constraints)
    print("========================================\n")

    # --------------------------------------------------------
    # MCP CALLS
    # --------------------------------------------------------

    airports = asyncio.run(
        list_airports(
            destination,
            limit=10,
        )
    )

    airlines = asyncio.run(
        list_airlines(
            "",
            limit=10,
        )
    )

    print("\n========== AIRPORT MCP DATA ==========")
    print(airports)
    print("======================================\n")

    print("\n========== AIRLINE MCP DATA ==========")
    print(airlines)
    print("======================================\n")

    # --------------------------------------------------------
    # LIMIT MCP DATA BEFORE SENDING TO LLM
    # --------------------------------------------------------

    airports_text = _limit_text(
        airports,
        1800,
    )

    airlines_text = _limit_text(
        airlines,
        1800,
    )

    prompt = f"""
Create concise flight guidance for this trip.

User request:
{query}

Trip constraints:
{constraints}

Airport MCP data:
{airports_text}

Airline MCP data:
{airlines_text}

Include:

1. Likely departure and arrival airports
2. Relevant airlines
3. Estimated flight duration
4. Approximate fare range
5. Peak-season warning
6. Booking advice

Keep the response concise.
Do not repeat large amounts of MCP data.
"""

    result = _llm_text(
        "You are a flight planning specialist. Give concise and practical answers.",
        prompt,
    )

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [
            AIMessage(
                content="Flight agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HOTEL AGENT
# ============================================================

def hotel_agent(state: TravelState):

    query = (
        f"Best hotels and areas to stay for: "
        f"{state['user_query']}"
    )

    print("\n========== HOTEL AGENT INPUT ==========")
    print(query)
    print("=======================================\n")

    # --------------------------------------------------------
    # TAVILY SEARCH
    # --------------------------------------------------------

    result = asyncio.run(
        tavily_search(query)
    )

    print("\n========== HOTEL SEARCH RESULT ==========")
    print(result)
    print("=========================================\n")

    # --------------------------------------------------------
    # LIMIT SEARCH RESULT
    # --------------------------------------------------------

    hotel_result_text = _limit_text(
        result,
        3500,
    )

    return {
        "hotel_results": hotel_result_text,
        "messages": [
            AIMessage(
                content="Hotel agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0),
    }


# ============================================================
# WEATHER AGENT
# ============================================================

def weather_agent(state: TravelState):

    constraints = state["trip_constraints"]
    city = constraints["destination"]

    print("\n========== WEATHER AGENT INPUT ==========")
    print("City:", city)
    print("=========================================\n")

    # --------------------------------------------------------
    # MCP WEATHER CALLS
    # --------------------------------------------------------

    weather_data = asyncio.run(
        current_weather(city)
    )

    forecast_data = asyncio.run(
        forecast(city)
    )

    print("\n========== CURRENT WEATHER ==========")
    print(weather_data)
    print("=====================================\n")

    print("\n========== WEATHER FORECAST ==========")
    print(forecast_data)
    print("======================================\n")

    # --------------------------------------------------------
    # LIMIT WEATHER DATA
    # --------------------------------------------------------

    weather_text = _limit_text(
        weather_data,
        1200,
    )

    forecast_text = _limit_text(
        forecast_data,
        2000,
    )

    result = f"""
Current weather:
{weather_text}

Forecast:
{forecast_text}
"""

    print("\n========== WEATHER AGENT OUTPUT ==========")
    print(result)
    print("==========================================\n")

    return {
        "weather_results": result,
        "messages": [
            AIMessage(
                content="Weather agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0),
    }


# ============================================================
# BUDGET AGENT
# ============================================================

def budget_agent(state: TravelState):

    print("\n========== BUDGET AGENT INPUT ==========")

    print("Trip Constraints:")
    print(
        state.get(
            "trip_constraints",
            {},
        )
    )

    print("\nFlight Results:")
    print(
        state.get(
            "flight_results",
            "",
        )
    )

    print("\nHotel Results:")
    print(
        state.get(
            "hotel_results",
            "",
        )
    )

    print("\nWeather Results:")
    print(
        state.get(
            "weather_results",
            "",
        )
    )

    print("=========================================\n")

    # --------------------------------------------------------
    # LIMIT PREVIOUS AGENT RESULTS
    # --------------------------------------------------------

    flight_data = _limit_text(
        state.get("flight_results", ""),
        1800,
    )

    hotel_data = _limit_text(
        state.get("hotel_results", ""),
        1800,
    )

    weather_data = _limit_text(
        state.get("weather_results", ""),
        1200,
    )

    constraints = _limit_text(
        state.get("trip_constraints", {}),
        800,
    )

    user_query = _limit_text(
        state["user_query"],
        1000,
    )

    # --------------------------------------------------------
    # BUDGET PROMPT
    # --------------------------------------------------------

    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User request:
{user_query}

Trip constraints:
{constraints}

Flight information:
{flight_data}

Hotel information:
{hotel_data}

Weather information:
{weather_data}

Provide a concise budget assessment containing:

1. Estimated cost categories
2. Main financial risk areas
3. Money-saving suggestions
4. Whether the trip appears feasible within the stated budget

Keep the response concise.
Do not repeat the source data.
"""

    result = _llm_text(
        "You are a practical travel budget analyst. Give concise, useful answers.",
        prompt,
    )

    print("\n========== BUDGET AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "budget_results": result,
        "messages": [
            AIMessage(
                content="Budget agent completed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# ITINERARY AGENT
# ============================================================

def itinerary_agent(state: TravelState):

    print("\n========== ITINERARY AGENT INPUT ==========")

    print("Trip Constraints:")
    print(
        state.get(
            "trip_constraints",
            {},
        )
    )

    print("\nFlight Results:")
    print(
        state.get(
            "flight_results",
            "",
        )
    )

    print("\nHotel Results:")
    print(
        state.get(
            "hotel_results",
            "",
        )
    )

    print("\nWeather Results:")
    print(
        state.get(
            "weather_results",
            "",
        )
    )

    print("\nBudget Results:")
    print(
        state.get(
            "budget_results",
            "",
        )
    )

    print("===========================================\n")

    # --------------------------------------------------------
    # LIMIT PREVIOUS AGENT RESULTS
    # --------------------------------------------------------

    flight_data = _limit_text(
        state.get("flight_results", ""),
        1500,
    )

    hotel_data = _limit_text(
        state.get("hotel_results", ""),
        1500,
    )

    weather_data = _limit_text(
        state.get("weather_results", ""),
        1000,
    )

    budget_data = _limit_text(
        state.get("budget_results", ""),
        1500,
    )

    constraints = _limit_text(
        state.get("trip_constraints", {}),
        800,
    )

    user_query = _limit_text(
        state["user_query"],
        1000,
    )

    # --------------------------------------------------------
    # ITINERARY PROMPT
    # --------------------------------------------------------

    prompt = f"""
Create a clear and practical draft travel itinerary.

User request:
{user_query}

Trip constraints:
{constraints}

Flight information:
{flight_data}

Hotel information:
{hotel_data}

Weather information:
{weather_data}

Budget assessment:
{budget_data}

Create a structured itinerary.

Include:

- Day-by-day plan
- Suggested activities
- Accommodation area
- Travel considerations
- Weather considerations
- Budget considerations

Keep the itinerary concise and practical.
Do not repeat the source data.
"""

    result = _llm_text(
        "You are an expert itinerary planner. Create concise, practical travel plans.",
        prompt,
    )

    print("\n========== ITINERARY OUTPUT ==========")
    print(result)
    print("======================================\n")

    # --------------------------------------------------------
    # HUMAN APPROVAL REQUEST
    # --------------------------------------------------------

    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HUMAN APPROVAL AGENT
# ============================================================

def human_approval_agent(state: TravelState):

    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get(
                "itinerary",
                "",
            ),
            "approval_request": state.get(
                "approval_request",
                "",
            ),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    approved = feedback["approved"]

    human_feedback = feedback["feedback"]

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================

def final_response_agent(state: TravelState):

    print("\n========== FINAL AGENT INPUT ==========")

    print(
        "Approved:",
        state.get("approved"),
    )

    print(
        "Feedback:",
        state.get("human_feedback"),
    )

    print("=======================================\n")

    # --------------------------------------------------------
    # LIMIT DATA
    # --------------------------------------------------------

    itinerary_data = _limit_text(
        state.get("itinerary", ""),
        3500,
    )

    budget_data = _limit_text(
        state.get("budget_results", ""),
        1500,
    )

    user_query = _limit_text(
        state.get("user_query", ""),
        1000,
    )

    feedback_data = _limit_text(
        state.get("human_feedback", ""),
        1000,
    )

    # --------------------------------------------------------
    # APPROVED
    # --------------------------------------------------------

    if state["approved"]:

        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

User request:
{user_query}

Draft itinerary:
{itinerary_data}

Budget notes:
{budget_data}

Return a clean, user-ready travel plan.

Keep it concise and well structured.
Do not mention internal agents or MCP.
"""

    # --------------------------------------------------------
    # NOT APPROVED
    # --------------------------------------------------------

    else:

        prompt = f"""
The human did not approve the draft itinerary.

Original user request:
{user_query}

Draft itinerary:
{itinerary_data}

Human feedback:
{feedback_data}

Budget notes:
{budget_data}

Produce an improved final travel plan that addresses
the human feedback.

Keep it concise and practical.
Do not mention internal agents or MCP.
"""

    result = _llm_text(
        "You produce final user-ready travel plans. Be concise and practical.",
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [
            AIMessage(
                content=result
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }