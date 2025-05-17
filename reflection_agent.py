from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
# from langchain_google_vertexai import ChatVertexAI
import json
import csv

load_dotenv()

# Load additional context from JSON files
with open("TEMP_story.json", "r", encoding="utf-8") as f:
    STORY_JSON_TEXT = json.dumps(json.load(f), indent=2)

with open("TEMP_entities.json", "r", encoding="utf-8") as f:
    ENTITIES_JSON_TEXT = json.dumps(json.load(f), indent=2)

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an entity relationship extractor that given a list of entities and a story text which is about "
            "those entities, extracts relationships between them. "
            "Generate a table which has the following columns: Entity, Relationship, Target Entity."
            "Remember that each extraction (row) can only have one entity, one relationship, and one target. The number of extractions is not limited."
            "If the user provides critique, respond with a revised version of your previous attempts. "
            "Don't include any instructions in your final result, only the table.",
        ),
        (
            "system",
            "Here is the story and entity list:\n\n"
            "Story context (JSON):\n{story_json}\n\n"
            "Entity list (JSON):\n{entities_json}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Freeze the JSON context into the prompt once
generate_prompt = generate_prompt.partial(
    story_json=STORY_JSON_TEXT,
    entities_json=ENTITIES_JSON_TEXT,
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an entity relationship extractor grading how good relationships between entities are extracted from a given story text. "
            "Generate critique and recommendations for the user's entity relationships."
            "There cannot be more than one entity, one relationship, and one target per set of extraction. Although the number of extractions is not limited."
            "Always provide detailed recommendations and reasons.",
        ),
        (
            "system",
            "Here is the story text:\n\n"
            "Story context (JSON):\n{story_json}\n\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = reflection_prompt.partial(
    story_json=STORY_JSON_TEXT,
    entities_json=ENTITIES_JSON_TEXT,
)

generating_llm = ChatOpenAI(temperature=0, model="gpt-4.1")
reflection_llm = ChatOpenAI(temperature=0, model="gpt-4.1")
# Update the ChatVertexAI configuration
# reflection_llm = ChatVertexAI(
#     model="gemini-2.5-pro-preview-03-25",
#     temperature=0,
#     # Add message formatting parameters if needed
#     convert_system_message_to_human=True  # This helps with system messages handling
# )
# model="gemini-2.5-pro-preview-03-25",

generate_chain = generate_prompt | generating_llm
reflect_chain = reflection_prompt | reflection_llm

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())
# print(graph.get_graph().print_ascii())

if __name__ == "__main__":
    print("Hello LangGraph!")
    # Seed the graph with an initial human message (can be empty or a task prompt)
    initial_messages = [
        HumanMessage(content="Please extract all relationships in the story.")
    ]
    # Pass the list of messages directly—LangGraph already routes it into the
    # MessagesPlaceholder inside generation_node
    response = graph.invoke(initial_messages)

    # Convert the LangChain messages to a serializable format
    serializable_response = []
    for message in response:
        serializable_response.append({
            "type": message.type,
            "content": message.content,
        })

    # Write the response to a JSON file
    with open("agent_output.json", "w", encoding="utf-8") as f:
        json.dump(serializable_response, f, indent=2)

    with open("agent_output2.json", "w", encoding="utf-8") as f:
        json.dump(response[-1].content, f, indent=2)

    # Parse the markdown table from the response and convert to CSV
    table_content = response[-1].content

    # Split by lines and filter out separator lines
    lines = [line.strip() for line in table_content.split('\n') if line.strip()]
    header_line = lines[0]
    data_lines = [line for line in lines[2:] if not line.startswith('|--')]

    # Extract column headers
    headers = [col.strip() for col in header_line.split('|') if col.strip()]

    # Extract data rows
    rows = []
    for line in data_lines:
        columns = [col.strip() for col in line.split('|') if col.strip()]
        rows.append(columns)

    # Write to CSV file
    with open("relationships.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        csv_writer.writerows(rows)

    print(f"Response has been written to agent_output.json")
    print(f"Relationships table has been written to relationships.csv")
    print(response)