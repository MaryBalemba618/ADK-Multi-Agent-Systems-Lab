"""
Defines the agents for the second part of the lab (workflow agents example).

This module builds a multi-agent system to generate a movie pitch.
It includes initial definitions for:
- 'researcher': An agent that uses Wikipedia to find facts.
- 'screenwriter': An agent that writes a plot outline.
- 'file_writer': An agent that saves the final pitch to a file.
- 'film_concept_team': A SequentialAgent that orchestrates the simple workflow.
- 'root_agent' ('greeter'): The parent agent that starts the user interaction.
- Helper tools: 'append_to_state' and 'write_file'.
"""
import os
import logging
import google.cloud.logging
from google.adk.tools import exit_loop

from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool  # import
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")
print(model_name)

# Tools


def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key.

    Args:
        field (str): a field name to append to
        response (str): a string to append to the field

    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}


def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(content)
    return {"status": "success"}


# Agents

file_writer = Agent(
    name="file_writer",
    model=model_name,
    description="Combines the plot and production reports into a saved file.",
    instruction="""
        Create a movie pitch document using the information below.

        PLOT_OUTLINE:
        {PLOT_OUTLINE?}

        BOX_OFFICE_REPORT:
        {box_office_report?}

        CASTING_REPORT:
        {casting_report?}

        INSTRUCTIONS:
        - Choose an appropriate movie title.
        - Use write_file to save the document.
        - Set directory to "movie_pitches".
        - Use a simple filename ending in ".txt".
        - Include these sections:
          1. Title and logline
          2. Final plot outline
          3. Box office assessment
          4. Casting suggestions
        - Use the latest plot outline if multiple drafts exist.
        - Preserve qualifications about unverified claims.
        - After the tool succeeds, tell the user the filename.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[write_file],
)

critic = Agent(
    name="critic",
    model=model_name,
    description="Reviews the movie outline and suggests improvements.",
    instruction="""
        Review the current plot using these questions:
        - Does it have a clear three-act structure?
        - Are the characters' struggles engaging?
        - Does the story fit its historical period?
        - Does it use the research effectively?

        If the plot meets these requirements, call exit_loop.

        If significant improvements are needed, use append_to_state
        to add your feedback to CRITICAL_FEEDBACK.

        Explain your decision briefly.

        PLOT_OUTLINE:
        {PLOT_OUTLINE?}

        RESEARCH:
        {research?}
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[append_to_state, exit_loop],
)

researcher = Agent(
    name="researcher",
    model=model_name,
    description="Answer research questions using Wikipedia.",
    instruction="""
    PROMPT:
    { PROMPT? }

    PLOT_OUTLINE:
    { PLOT_OUTLINE? }

    CRITICAL_FEEDBACK:
    { CRITICAL_FEEDBACK? }

    INSTRUCTIONS:
    - If there is a CRITICAL_FEEDBACK, use your wikipedia tool to do research to solve those suggestions
    - If there is a PLOT_OUTLINE, use your wikipedia tool to do research to add more historical detail
    - If these are empty, use your Wikipedia tool to gather facts about the person in the PROMPT
    - Use the 'append_to_state' tool to add your research to the field 'research'.
    - Summarize what you have learned.
    Now, use your Wikipedia tool to do research.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)
screenwriter = Agent(
    name="screenwriter",
    model=model_name,
    description="Writes and improves a historical movie plot.",
    instruction="""
        Write a movie pitch based on the user's subject and research.

        USER SUBJECT:
        {PROMPT?}

        RESEARCH:
        {research?}

        PREVIOUS PLOT:
        {PLOT_OUTLINE?}

        CRITIC FEEDBACK:
        {CRITICAL_FEEDBACK?}

        INSTRUCTIONS:
        - Create a title, logline, and three-act plot outline.
        - Use historical details from the research.
        - Clearly distinguish fictional story elements from facts.
        - If critic feedback exists, revise the previous plot
          to address that feedback.
        - Call append_to_state with field="PLOT_OUTLINE"
          and response containing the complete updated pitch.
        - Briefly summarize what you wrote or changed.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[append_to_state],
)
writers_room = LoopAgent(
    name="writers_room",
    description="Refines a historical movie plot through research and review.",
    sub_agents=[
        researcher,
        screenwriter,
        critic,
    ],
    max_iterations=5,
)
box_office_researcher = Agent(
    name="box_office_researcher",
    model=model_name,
    description="Discusses the film's potential audience and commercial appeal.",
    instruction="""
        PLOT_OUTLINE:
        {PLOT_OUTLINE?}

        Discuss the potential audience, genre, commercial strengths,
        and financial risks of this movie.

        Treat this as a preliminary assessment.
        Do not invent box office figures or present estimates as facts.
        Clearly label any comparisons that need verification.
    """,
    output_key="box_office_report",
)

casting_agent = Agent(
    name="casting_agent",
    model=model_name,
    description="Suggests actors for the movie's main characters.",
    instruction="""
        PLOT_OUTLINE:
        {PLOT_OUTLINE?}

        Suggest actors for the main characters.
        Explain how each actor could fit the role.
        These are creative suggestions, not confirmed casting.
    """,
    output_key="casting_report",
)

preproduction_team = ParallelAgent(
    name="preproduction_team",
    sub_agents=[
        box_office_researcher,
        casting_agent,
    ],
)
film_concept_team = SequentialAgent(
    name="film_concept_team",
    description="Create a movie pitch with production reports and save it.",
    sub_agents=[
        writers_room,
        preproduction_team,
        file_writer,
    ],
)

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="Guides the user in crafting a movie plot.",
    instruction="""
    - Let the user know you will help them write a pitch for a hit movie. Ask them for   
      a historical figure to create a movie about.
    - When they respond, use the 'append_to_state' tool to store the user's response
      in the 'PROMPT' state key and transfer to the 'film_concept_team' agent
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[append_to_state],
    sub_agents=[film_concept_team],
)