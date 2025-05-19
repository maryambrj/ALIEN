from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
import json
import csv
import math

load_dotenv()

def load_entity_story_rows_from_csv(csv_path: str):
    """
    Read CSV file where:
      - The first column is 'entity',
      - The second column is 'target entity',
      - The last column is 'story text'.
    Returns a tuple: (header, list of dicts for each (entity, target, story)).
    """
    rows = []
    header = None
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # extract header as the first thing
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
    return header, rows

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

def chunk_list(lst, n_chunks):
    """Yield successive n_chunks pieces from lst as evenly as possible."""
    k, m = divmod(len(lst), n_chunks)
    for i in range(n_chunks):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]

CSV_PATH = "test_refined_filtered_lite.csv"
CHUNKS = 2  # Number of chunks; can make this a parameter as needed

# Remove header from input as the first thing
input_header, ENTITY_STORY_ROWS = load_entity_story_rows_from_csv(CSV_PATH)
chunks = list(chunk_list(ENTITY_STORY_ROWS, CHUNKS))

generate_prompt_template = [
    (
        "system",
        "You are an entity relationship extractor that, given entity, target entity, and their corresponding story text, "
        "extracts relationships between the entity and target entity. "
        "Generate a table with the columns: Entity, Relationship, Target Entity. "
        "Each extraction (row) should have one entity, one relationship, and one target corresponding to the story provided. "
        "Relations MUST be one of these: 'no_relation', 'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of', 'acquired_on', "
        "'agreement_with', 'operations_in', 'subsidiary_of','employee_of', 'attended', 'cost_of', 'acquired_by', 'member_of', 'profit_of', "
        "'revenue_of', 'founder_of', 'formed_on'."
        "If you don't find any relation, answer with 'no_relation'. Otherwise, choose the most appropriate relation among the ones mentioned above. "
        "Direction matters: relationship is from Entity to Target Entity (do not include both sides). "
        "Remember to process all rows (the entire table)."
        "Always separate the columns with a pipe '|' in your table to keep the format consistent."
        "Don't extract relationships for any entity not listed."
        "Always include the headers | Entity | Target Entity | Story Text | as the first row of the table. "
        "If the user provides critique, respond with a revised version. No instructions in final result - only the table."
    ),
    (
        "system",
        "Here are the entities, target entities, and corresponding story texts:\n\n"
        "{entity_story_context}\n"
    ),
    MessagesPlaceholder(variable_name="messages"),
]

reflection_prompt_template = [
    (
        "system",
        "You are an entity relationship extractor grading the extractions in a table, given the entities, target entities, and the corresponding story text. "
        "Generate critique and recommendations about the quality of extracted relationships. "
        "The relations CANNOT be anything but one of these: 'no_relation', 'headquartered_in', 'formed_in', 'title', 'shares_of', 'loss_of', "
        "'acquired_on', 'agreement_with', 'operations_in', 'subsidiary_of','employee_of', 'attended', 'cost_of', 'acquired_by', 'member_of', "
        "'profit_of', 'revenue_of', 'founder_of', 'formed_on'"
        "Make sure the columns are separated with a pipe '|' in the table and not comma ','."
        "No more than one entity, relationship, and target per extraction. Always provide detailed critique."
    ),
    (
        "system",
        "Here is the context (table):\n\n"
        "{entity_story_context}\n"
    ),
    MessagesPlaceholder(variable_name="messages"),
]

generating_llm = ChatOpenAI(model="o3-mini-2025-01-31")
reflection_llm = ChatOpenAI(temperature=0, model="gpt-4o")

# Agent execution for each chunk
def run_agent_on_chunk(chunk_rows):
    entity_story_context = make_context_table(chunk_rows)
    generate_prompt = ChatPromptTemplate.from_messages(generate_prompt_template).partial(
        entity_story_context=entity_story_context,
    )
    reflection_prompt = ChatPromptTemplate.from_messages(reflection_prompt_template).partial(
        entity_story_context=entity_story_context,
    )
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
    initial_messages = [
        HumanMessage(content="Please extract all relationships in the stories provided.")
    ]
    response = graph.invoke(initial_messages)
    return response

def parse_markdown_table(table_content):
    """
    Parses a markdown table string.

    Returns:
        headers (list): List of column headers (or None, if not found)
        rows (list): List of lists, each being a row of cell values (excluding header & separator)
    """
    lines = [line.strip() for line in table_content.split('\n') if line.strip()]
    # Filter out separator lines (e.g., |---|---|---|)
    not_sep = lambda line: not set(line.replace('|', '').strip()).issubset({'-', ':'})
    lines = [line for line in lines if line.startswith('|') and line.endswith('|')]

    header = None
    rows = []
    for idx, line in enumerate(lines):
        # Remove the pipes at front and end, then split and strip
        cols = [cell.strip() for cell in line.strip('|').split('|')]
        if idx == 0:
            header = cols
        elif idx == 1 and not_sep(line):
            # Sometimes the separator is NOT using only dashes, which we cautiously retain as data
            rows.append(cols)
        elif idx > 1:
            rows.append(cols)
    # Remove the separator line if present (usually second line, with only - or :)
    if rows and all(set(cell) <= {'-', ':'} for cell in rows[0]):
        rows = rows[1:]
    return header, rows

if __name__ == "__main__":
    print("Hello LangGraph! Running chunked agent extraction...")

    all_rows = []
    # Use your desired/fixed header order here
    csv_headers = ["head_entity_text", "tail_entity_text", "relation"]
    for ix, chunk in enumerate(chunks):
        print(f"Processing chunk {ix+1}/{CHUNKS} (rows {len(chunk)}) ...")
        response = run_agent_on_chunk(chunk)
        table_content = response[-1].content
        headers, rows = parse_markdown_table(table_content)
        # Map the extracted row columns to fixed header order
        # headers: [Entity, Target Entity, Relationship] (after column swap)
        
        # Column indices after swap:
        # headers[0]: Entity -> head_entity_text
        # headers[1]: Target Entity -> tail_entity_text
        # headers[2]: Relationship -> relation
        
        for row in rows:
            # Guard against incomplete rows
            if len(row) >= 3:
                mapped_row = [row[0], row[1], row[2]]
                all_rows.append(mapped_row)
        # Optionally, store separate outputs per chunk for debugging/tracing
        with open(f"agent_output_chunk{ix+1}.json", "w", encoding="utf-8") as f:
            json.dump([{"type": m.type, "content": m.content} for m in response], f, indent=2)

    # At the END: add header for output CSV
    with open("relationships.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)
        csv_writer.writerows(all_rows)

    print(f"Combined relationships table has been written to relationships.csv")