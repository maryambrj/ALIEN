from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
# from langchain_google_vertexai import ChatVertexAI
# from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph
from pydantic import BaseModel, Field
from typing import List, Literal, Sequence
import pandas as pd
import json
import csv
import math

load_dotenv()
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def load_entity_story_rows_from_csv(csv_path: str):
    """
    Read CSV file where:
      - The column with header 'head_entity_text' is treated as 'entity',
      - The column with header 'tail_entity_text' is treated as 'target entity',
      - The column with header 'sentence' is treated as 'story text'.
    Returns a tuple: (header, list of dicts for each (entity, target, story)).
    """

    rows = []
    header = None
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # extract header as the first thing

        # Find column indexes by header names
        if header is None:
            raise ValueError("CSV file is empty or missing header row.")

        try:
            entity_idx = header.index("entity1")
            target_entity_idx = header.index("entity2")
            story_text_idx = header.index("sentence")
        except ValueError as e:
            raise ValueError("CSV header missing required column(s): 'entity1', 'entity2', 'sentence'") from e

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
#########################################################################################
CSV_PATH = "test_semeval.csv"
CHUNKS = 2717  # Number of chunks; can make this a parameter as needed
#########################################################################################

# Remove header from input as the first thing
input_header, ENTITY_STORY_ROWS = load_entity_story_rows_from_csv(CSV_PATH)
chunks = list(chunk_list(ENTITY_STORY_ROWS, CHUNKS))

# generate_prompt_template = [
#     (
#         "system",
#         "You are an entity relationship extractor that, given entity, target entity, and their corresponding story text, "
#         "extracts relationships between the entity and target entity. "
#         "Generate a table with the columns: Entity, Target Entity, and Relation. Do not use any other information than the story text."
#         "Each extraction (row) should have one entity, one relationship, and one target corresponding to the story provided."
#         "Relations MUST be one of these: 'Message-Topic (e1, e2)','Product-Producer (e2, e1)','Instrument-Agency (e2, e1)','Entity-Destination (e1, e2)','Cause-Effect (e2, e1)',"
#         "'Component-Whole (e1, e2)', 'Product-Producer (e1, e2)', 'Member-Collection (e2, e1)', 'Other', 'Entity-Origin (e1, e2)', 'Content-Container (e1, e2)',"
#         "'Entity-Origin (e2, e1)', 'Cause-Effect (e1, e2)', 'Component-Whole (e2, e1)', 'Content-Container (e2, e1)', 'Instrument-Agency (e1, e2)', 'Message-Topic (e2, e1)',"
#         "'Member-Collection (e1, e2)', 'Entity-Destination (e2, e1)'."
#         "Choose the closest option among the ones mentioned above as relation between entity and target entity. If you can't find any relation given the story text, answer with 'Other'. "
#         "Direction matters: relationship is from Entity to Target Entity (do not include both sides). If the relation is from entity to target entity, choose then (e1, e2) version, or if vice versa choose (e2, e1)"
#         "Remember to process all rows (the entire table). So the number of input data rows and output must be the same."
#         "Always separate the columns with a pipe '|' in your table to keep the format consistent."
#         "Don't extract relationships for any entity not listed."
#         "Always include the headers | Entity | Target Entity | Relation | as the first row of the table. "
#         "If the user provides critique, respond with a revised version. No instructions in final result - only the table."
#     ),
#     (
#         "system",
#         "Here are the entities, target entities, and corresponding story texts:\n\n"
#         "{entity_story_context}\n"
#     ),
#     MessagesPlaceholder(variable_name="messages"),
# ]

# reflection_prompt_template = [
#     (
#         "system",
#         "You are an entity relationship extractor that finds relationship, given the entities, target entities, and the corresponding story text. "
#         "Generate critique and recommendations about the quality of extracted relationships. "
#         "The relations CANNOT be anything but one of these:"
#         "'Message-Topic (e1, e2)','Product-Producer (e2, e1)','Instrument-Agency (e2, e1)','Entity-Destination (e1, e2)','Cause-Effect (e2, e1)',"
#         "'Component-Whole (e1, e2)', 'Product-Producer (e1, e2)', 'Member-Collection (e2, e1)', 'Other', 'Entity-Origin (e1, e2)', 'Content-Container (e1, e2)',"
#         "'Entity-Origin (e2, e1)', 'Cause-Effect (e1, e2)', 'Component-Whole (e2, e1)', 'Content-Container (e2, e1)', 'Instrument-Agency (e1, e2)', 'Message-Topic (e2, e1)',"
#         "'Member-Collection (e1, e2)', 'Entity-Destination (e2, e1)'"
#         "Make sure the columns are separated with a pipe '|' in the table and not comma ','."
#         "Count the number of rows in the original table and compare with the number of rows in the output table. If not the same, point out which rows the output table is missing to be added."
#         "If the relation is from entity to target entity, choose then (e1, e2) version, or if vice versa choose (e2, e1)."
#         "Pay special attention to 'Other' answers and review if there is truly no relation between entity, target entity, given the story text. "
#         "No more than one entity, target, and relationship per extraction per given story. Allow duplicates. Always provide detailed critique."
#     ),
#     (
#         "system",
#         "Here is the context (table):\n\n"
#         "{entity_story_context}\n"
#     ),
#     MessagesPlaceholder(variable_name="messages"),
# ]


class TableRow(BaseModel):
    head_entity_text: str
    tail_entity_text: str
    relation: Literal[
        "Message-Topic (e1, e2)","Product-Producer (e2, e1)","Instrument-Agency (e2, e1)","Entity-Destination (e1, e2)","Cause-Effect (e2, e1)",
        "Component-Whole (e1, e2)", "Product-Producer (e1, e2)", "Member-Collection (e2, e1)", "Other", "Entity-Origin (e1, e2)", "Content-Container (e1, e2)",
        "Entity-Origin (e2, e1)", "Cause-Effect (e1, e2)", "Component-Whole (e2, e1)", "Content-Container (e2, e1)", "Instrument-Agency (e1, e2)", "Message-Topic (e2, e1)",
        "Member-Collection (e1, e2)", "Entity-Destination (e2, e1)"]


class LLMTableOutput(BaseModel):
    headers: List[str]  # should be ["name", "age", "status"]
    rows: List[TableRow]


# output_parser = PydanticOutputParser(pydantic_object=LLMTableOutput)

# generating_llm_raw = ChatVertexAI(temperature = 0, model="gemini-2.5-pro-preview-05-06")
# generating_llm_raw = ChatOpenAI(model="o3-mini-2025-01-31")
# generating_llm_raw = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     max_tokens=None,
#     timeout=None
# )
# generating_llm_raw = ChatGoogleGenerativeAI(temperature = 0, model="gemini-2.5-flash-preview-05-20")
generating_llm_raw = ChatOpenAI(temperature=0, model="gpt-4o")

generating_llm = generating_llm_raw.with_structured_output(LLMTableOutput)

# generating_llm = ChatOpenAI(model="o3-mini-2025-01-31")
# reflection_llm = ChatOpenAI(temperature=0, model="gpt-4o")

# Agent execution for each chunk
def run_agent_on_chunk(chunk_rows):



    entity_story_context = make_context_table(chunk_rows)

    generate_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an entity relationship extractor that, given entity, target entity, and their corresponding story text, "
            "extracts relationships between the entity and target entity. "
            "Generate a table with the columns: Entity, Target Entity, and Relation. Do not use any other information than the story text."
            "Each extraction (row) should have one entity, one relationship, and one target corresponding to the story provided. "
            "Relations MUST be one of these: 'Message-Topic (e1, e2)','Product-Producer (e2, e1)','Instrument-Agency (e2, e1)','Entity-Destination (e1, e2)','Cause-Effect (e2, e1)',"
            "'Component-Whole (e1, e2)', 'Product-Producer (e1, e2)', 'Member-Collection (e2, e1)', 'Other', 'Entity-Origin (e1, e2)', 'Content-Container (e1, e2)',"
            "'Entity-Origin (e2, e1)', 'Cause-Effect (e1, e2)', 'Component-Whole (e2, e1)', 'Content-Container (e2, e1)', 'Instrument-Agency (e1, e2)', 'Message-Topic (e2, e1)',"
            "'Member-Collection (e1, e2)', 'Entity-Destination (e2, e1)'."
            "If you don't find any relation, answer with 'Other'. Otherwise, choose the most appropriate relation among the ones mentioned above. "
            "Direction matters: relationship is from Entity to Target Entity (do not include both sides). "
            "Remember to process all rows (the entire table)."
            "Always separate the columns with a pipe '|' in your table to keep the format consistent."
            "Don't extract relationships for any entity not listed."
            "If the relation is from entity to target entity, choose then (e1, e2) version, or if vice versa choose (e2, e1)."
            "Always include the headers | Entity | Target Entity | Relation | as the first row of the table. "
            "If the user provides critique, respond with a revised version. No instructions in final result - only the table."
            "Here are some examples with answer:"
            "in story text 'The team stapled the plastic along the joists with heavy duty staple guns to hold it in place.' the relation between 'team' (e1) and 'gun' (e2) is 'Instrument-Agency (e2, e1)'."
            "in story text 'The barbels of the exposed catfish curled within 2 hours in heptachlor.' the relation between 'barbels' (e1) and 'catfish' (e2) is 'Component-Whole (e1, e2)'."
            "in story text 'The confrontation between Martyn and Liz earlier ended with a struggle that resulted in Martyn's drowning.' the relation between 'struggle' (e1) and 'drowning' (e2) is 'Cause-Effect (e1, e2)'."
            "in story text 'The opening angle of the Cherenkov cone measures the velocity with a resolution of 0.2%.' the relation between 'angle' (e1) and 'resolution' (e2) is 'Other'."
        ),
        (
            "system",
            "Here are the entities, target entities, and corresponding story texts:\n\n"
            "{entity_story_context}\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(
        entity_story_context=entity_story_context,
    )

    # reflection_prompt = ChatPromptTemplate.from_messages(reflection_prompt_template).partial(
    #     entity_story_context=entity_story_context,
    # )
    generate_chain = generate_prompt | generating_llm
    # reflect_chain = reflection_prompt | reflection_llm
    # REFLECT = "reflect"
    GENERATE = "generate"


    def generation_node(state: Sequence[BaseMessage]):
        output = generate_chain.invoke({"messages": state})
        # Ensure the output is always wrapped as a message (convert to readable string or format as needed)
        temp = [row.model_dump() for row in output.rows]
        return [AIMessage(content=json.dumps(temp))]

    def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        res = reflect_chain.invoke({"messages": messages})
        return [HumanMessage(content=res.content)]


    builder = MessageGraph()
    builder.add_node(GENERATE, generation_node)
    # builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(GENERATE)

    # def should_continue(state: List[BaseMessage]):
    #     if len(state) > 6:
    #         return END
    #     return REFLECT

    # builder.add_conditional_edges(GENERATE, should_continue)
    # builder.add_edge(REFLECT, GENERATE)
    graph = builder.compile()
    initial_messages = [
        HumanMessage(content="Please extract all relationships in the stories provided.")
    ]
    response = graph.invoke(initial_messages)
    return response

if __name__ == "__main__":
    print("Hello LangGraph! Running chunked agent extraction...")

    all_data = []
    all_rows = []
    # Use your desired/fixed header order here
    csv_headers = ["head_entity_text", "tail_entity_text", "relation"]
    for ix, chunk in enumerate(chunks):
        print(f"Processing chunk {ix+1}/{CHUNKS} (rows {len(chunk)}) ...")
        response = run_agent_on_chunk(chunk)
        table_content = response[-1].content


        data = json.loads(table_content)  # This gives you a list of dicts
        all_data.extend(data)



        # Map the extracted row columns to fixed header order
        # headers: [Entity, Target Entity, Relationship] (after column swap)
        
        # Column indices after swap:
        # headers[0]: Entity -> head_entity_text
        # headers[1]: Target Entity -> tail_entity_text
        # headers[2]: Relationship -> relation

    df = pd.DataFrame(all_data)
    df.to_csv('relationships_semeval_fewshot4O.csv', index=False)

    # At the END: add header for output CSV

    print(f"Combined relationships table has been written to relationships_semeval_fewshot4O.csv")