"""
Defines the agents for the first part of the lab (parent-subagent example).

This module contains the initial definitions for:
- 'attractions_planner': A sub-agent to list attractions for a country.
- 'travel_brainstormer': A sub-agent to help a user decide on a country.
- 'root_agent' ('steering'): The parent agent that directs the conversation
                            to the correct sub-agent.
"""
import os
import sys
import logging

sys.path.append("..")
from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv
import google.cloud.logging
from google.adk import Agent
from google.genai import types
from typing import Optional, List, Dict

from google.adk.tools.tool_context import ToolContext

load_dotenv()

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

# Tools (add the tool here when instructed)

def save_attractions_to_state(
tool_context: ToolContext,
attractions: List[str]
) -> dict[str, str]:
    """Saves the list of attractions to state["attractions"].

    Args:
        attractions [str]: a list of strings to add to the list of attractions

    Returns:
        None
    """
    # Load existing attractions from state. If none exist, start an empty list
    existing_attractions = tool_context.state.get("attractions", [])

    # Update the 'attractions' key with a combo of old and new lists.
    # When the tool is run, ADK will create an event and make
    # corresponding updates in the session's state.
    tool_context.state["attractions"] = existing_attractions + attractions

    # A best practice for tools is to return a status message in a return dict
    return {"status": "success"}
# Agents

attractions_planner = Agent(    
    name="attractions_planner",
    model=os.getenv("MODEL"),
    description="Build a list of attractions to visit in a country.",
    instruction="""
        - Provide the user options for attractions to visit within their selected country.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    # When instructed to do so, paste the tools parameter below this line
        Keep your existing attraction-planning instructions here.

        - When the user selects an attraction, use
          save_attractions_to_state to save it.
        - If they ask to view their list, provide a bulleted
          list of {attractions?} and suggest more attractions.
    """,
      tools=[save_attractions_to_state],

    )

travel_brainstormer = Agent(
    name="travel_brainstormer",
    model=os.getenv("MODEL"),
    description="Help a user decide what country to visit.",
    instruction="""
        Provide a few suggestions of popular countries for travelers.

        Help a user identify their primary goals of travel:
        adventure, leisure, learning, shopping, or viewing art

        Identify countries that would make great destinations
        based on their priorities.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

root_agent = Agent(
    name="steering",
    model=os.getenv("MODEL"),
    description="Start a user on a travel adventure.",
    instruction="""
        Ask the user if they know where they'd like to travel
        or if they need some help deciding.

        If they need help deciding, transfer to travel_brainstormer.
        If they have chosen a country, transfer to attractions_planner.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    sub_agents=[travel_brainstormer, attractions_planner],
)