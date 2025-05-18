from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
import json
import csv

load_dotenv()

# --- New: Read from CSV instead of JSON files ---

def load_entity_story_rows_from_csv(csv_path: str):
    """
    Read CSV file where:
      - The first column is 'entity',
      - The second column is 'target entity',
      - The last column is 'story text'.
    Returns a list of dicts for each (entity, target, story).
    """
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if not row or len(row) < 3:
                continue
            entity = row[0]
            target_entity = row[1]
            story_text = row[-1]
            rows.append(
                {
                    "entity": entity,
                    "target_entity": target_entity,
                    "story_text": story_text,
                }
            )
    return rows

CSV_PATH = "test_refined_filtered_lite.csv"
ENTITY_STORY_ROWS = load_entity_story_rows_from_csv(CSV_PATH)

# Prepare prompt context as text (for LLM)
def make_context_table(entity_story_rows):
    """Format the entity-target-story triplets as a table for LLm input."""
    table = "| Entity | Target Entity | Story Text |\n|---|---|---|\n"
    for row in entity_story_rows:
        # Make story and entity text compact (truncate if necessary)
        e = row["entity"]
        t = row["target_entity"]
        s = row["story_text"].replace("\n", " ").strip()
        s = s[:500] + "..." if len(s) > 500 else s
        table += f"| {e} | {t} | {s} |\n"
    return table

ENTITY_STORY_CONTEXT = make_context_table(ENTITY_STORY_ROWS)

# ----- Update the prompts -----

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an entity relationship extractor that, given entity, target entity, and their corresponding story text, "
            "extracts relationships between the entity and target entity. "
            "Generate a table with the columns: Entity, Relationship, Target Entity. "
            "Each extraction (row) should have one entity, one relationship, and one target corresponding to the story provided. "
            "Relations MUST be one of these: 'no_relation', 'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of', 'acquired_on', 'agreement_with', 'operations_in', 'subsidiary_of','employee_of', 'attended', 'cost_of', 'acquired_by', 'member_of', 'profit_of', 'revenue_of', 'founder_of', 'formed_on'."
            "If you don't find any relation, answer with 'no_relation'. Otherwise, choose the most appropriate relation among the ones mentioned above. "
            "Direction matters: relationship is from Entity to Target Entity (do not include both sides). "
            "Remember to process all rows (the entire table)."
            "Don't extract relationships for any entity not listed."
            "If the user provides critique, respond with a revised version. No instructions in final result - only the table."
        ),
        (
            "system",
            "Here are the entities, target entities, and corresponding story texts:\n\n"
            "{entity_story_context}\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
generate_prompt = generate_prompt.partial(
    entity_story_context=ENTITY_STORY_CONTEXT,
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an entity relationship extractor grading the extractions in a table, given the entities, target entities, and the corresponding story text. "
            "Generate critique and recommendations about the quality of extracted relationships. "
            "The relations CANNOT be anything but one of these: 'no_relation', 'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of', 'acquired_on', 'agreement_with', 'operations_in', 'subsidiary_of','employee_of', 'attended', 'cost_of', 'acquired_by', 'member_of', 'profit_of', 'revenue_of', 'founder_of', 'formed_on'"
            "No more than one entity, relationship, and target per extraction. Always provide detailed critique."
        ),
        (
            "system",
            "Here is the context (table):\n\n"
            "{entity_story_context}\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflection_prompt = reflection_prompt.partial(
    entity_story_context=ENTITY_STORY_CONTEXT,
)

generating_llm = ChatOpenAI(model="o3-mini-2025-01-31")
reflection_llm = ChatOpenAI(temperature=0, model="gpt-4o")

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

if __name__ == "__main__":
    print("Hello LangGraph!")
    initial_messages = [
        HumanMessage(content="Please extract all relationships in the stories provided.")
    ]
    response = graph.invoke(initial_messages)

    # Convert the LangChain messages to a serializable format
    serializable_response = []
    for message in response:
        serializable_response.append({
            "type": message.type,
            "content": message.content,
        })
    with open("agent_output.json", "w", encoding="utf-8") as f:
        json.dump(serializable_response, f, indent=2)

    with open("agent_output2.json", "w", encoding="utf-8") as f:
        json.dump(response[-1].content, f, indent=2)

    # Parse the markdown table from the response and convert to CSV
    table_content = response[-1].content
    lines = [line.strip() for line in table_content.split('\n') if line.strip()]
    header_line = lines[0]
    data_lines = [line for line in lines[2:] if not line.startswith('|--')]
    headers = [col.strip() for col in header_line.split('|') if col.strip()]
    rows = []
    for line in data_lines:
        columns = [col.strip() for col in line.split('|') if col.strip()]
        rows.append(columns)
    with open("relationships.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        csv_writer.writerows(rows)

    print(f"Response has been written to agent_output.json")
    print(f"Relationships table has been written to relationships.csv")
    print(response)