from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from pydantic import BaseModel, Field
from typing import List, Literal, Sequence
import pandas as pd
import json
import csv

load_dotenv()




def load_entity_story_rows_from_csv(csv_path: str):

    rows = []
    header = None
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)

        if header is None:
            raise ValueError("CSV file is empty or missing header row.")

        try:
            entity_idx = header.index("e1_name")
            target_entity_idx = header.index("e2_name")
            story_text_idx = header.index("context")
        except ValueError as e:
            raise ValueError("CSV header missing required column(s): 'e1_name', 'e2_name', 'sentence'") from e

        for row in reader:
            if not row or len(row) <= max(entity_idx, target_entity_idx, story_text_idx):
                continue  # skip incomplete rows
            entity = row[entity_idx]
            target_entity = row[target_entity_idx]
            story_text = row[story_text_idx]
            rows.append(
                {
                    "entity": entity,
                    "target_entity": target_entity,
                    "story_text": story_text,
                }
            )
    return header, rows

def make_context_table(entity_story_rows):
    table = "| Entity | Target Entity | Story Text |\n|---|---|---|\n"
    for row in entity_story_rows:
        e = row["entity"]
        t = row["target_entity"]
        s = row["story_text"].replace("\n", " ").strip()
        s = s[:500] + "..." if len(s) > 500 else s
        table += f"| {e} | {t} | {s} |\n"
    return table

def chunk_list(lst, n_chunks):
    k, m = divmod(len(lst), n_chunks)
    for i in range(n_chunks):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]
#########################################################################################
CSV_PATH = "./datasets/core_data/test_core.csv"
CHUNKS = 708
#########################################################################################

input_header, ENTITY_STORY_ROWS = load_entity_story_rows_from_csv(CSV_PATH)
chunks = list(chunk_list(ENTITY_STORY_ROWS, CHUNKS))

generate_prompt_template = [
    (
        "system",
        "You are an entity relationship extractor that, given entity, target entity, and their corresponding story text, "
        "extracts relationships and the direction of the relationship as a binary flag that has only 0 or 1. 0 means direction of the relationship is from entity to target entity." 
        "1 means direction of the relationship is from target entity to entity."
        "Generate a table with the columns: Entity, Target Entity, Relation, and Invert Relation Flag. Do not use any other information than the story text."
        "Each extraction (row) should have one entity, one relationship, the relationship direction, and one target corresponding to the story provided. "
        "Relations MUST be one of these: 'undefined', 'product_or_service_of', 'shareholder_of', 'collaboration', 'subsidiary_of'," 
        "'client_of', 'competitor_of', 'acquired_by', 'traded_on', 'regulated_by', 'brand_of', 'merged_with'."
        "Choose the closest option among the ones mentioned above as relation between entity and target entity. If you can't find any relation given the story text, answer with 'undefined'. "
        "Direction matters: if relation is from Entity to Target Entity set invert_relation to 0, otherwise if relation is from Target Entity to Entity then set invert_relation to 1. "
        "Remember to process all rows (the entire table). So the number of input data rows and output must be the same."
        "Always separate the columns with a pipe '|' in your table to keep the format consistent."
        "Don't extract relationships for any entity not listed."
        "Always include the headers | Entity | Target Entity | Relation | Invert Relation | as the first row of the table. "
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
        "You are an entity relationship extractor that finds relationship and invert_relation (i.e., 1 means from target entity to entity), given the entities, target entities, and the corresponding story text. "
        "Generate critique and recommendations about the quality of extracted relationships. "
        "The relations CANNOT be anything but one of these: 'undefined', 'product_or_service_of', 'shareholder_of', 'collaboration', 'subsidiary_of'," 
        "'client_of', 'competitor_of', 'acquired_by', 'traded_on', 'regulated_by', 'brand_of', 'merged_with'."
        "Make sure the columns are separated with a pipe '|' in the table and not comma ','."
        "Count the number of rows in the original table and compare with the number of rows in the output table. If not the same, point out which rows the output table is missing to be added."
        "Pay special attention to 'undefined' answers and review if there is truly no relation between entity, target entity, given the story text. "
        "Keep your feedback short and concise. Do not provide feedback on correctly identified relationships."
        "No more than one entity, target, and relationship and invert_relation per extraction per given story. Allow duplicates. Keep it concise."
    ),
    (
        "system",
        "Here is the context (table):\n\n"
        "{entity_story_context}\n"
    ),
    MessagesPlaceholder(variable_name="messages"),
]


class TableRow(BaseModel):
    entity: str
    target_entity: str
    relation: Literal[
        "undefined", "product_or_service_of", "shareholder_of", "collaboration", "subsidiary_of",
        "client_of", "competitor_of", "acquired_by", "traded_on", "regulated_by", "brand_of", "merged_with"]
    invert_relation: Literal["0", "1"] = Field(
        ...,
        description="Binary flag that accepts only 0 or 1. 0 means direction of the relationship is from e1 to e2. 1 means direction of the relationship is from e2 to e1."
    )

class LLMTableOutput(BaseModel):
    headers: List[str]
    rows: List[TableRow]


generating_llm_raw = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

generating_llm = generating_llm_raw.with_structured_output(LLMTableOutput)
reflection_llm = ChatOpenAI(model="gpt-4o")

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
        output = generate_chain.invoke({"messages": state})
        temp = [row.model_dump() for row in output.rows]
        return [AIMessage(content=json.dumps(temp))]

    def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        res = reflect_chain.invoke({"messages": messages})
        return [HumanMessage(content=[{"type": "text", "text": res.content}])]



    builder = MessageGraph()
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(GENERATE)

    def should_continue(state: List[BaseMessage]):
        if len(state) > 3:
            return END
        return REFLECT

    builder.add_conditional_edges(GENERATE, should_continue)
    builder.add_edge(REFLECT, GENERATE)
    graph = builder.compile()
    print(graph.get_graph().draw_mermaid())
    print(graph.get_graph().print_ascii())
    initial_messages = [
        HumanMessage(content=[{"type": "text", "text": "Please extract all relationships in the stories provided."}])
    ]
    response = graph.invoke(initial_messages)
    return response

if __name__ == "__main__":
    print("Running chunked agent extraction...")

    all_data = []
    all_rows = []
    csv_headers = ["e1_name", "e2_name", "relation", "invert_relation"]
    for ix, chunk in enumerate(chunks):
        print(f"Processing chunk {ix+1}/{CHUNKS} (rows {len(chunk)}) ...")
        response = run_agent_on_chunk(chunk)
        table_content = response[-1].content
        data = json.loads(table_content)  # This gives you a list of dicts
        all_data.extend(data)


    df = pd.DataFrame(all_data)
    df.to_csv('relationships_core.csv', index=False)


    print(f"Combined relationships table has been written to relationships_core.csv")