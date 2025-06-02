import os
import csv
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from sklearn.metrics import accuracy_score, f1_score
from pydantic import BaseModel, Field
from typing import Optional
import re
from langchain_core.tracers import LangChainTracer

load_dotenv()
LangChainTracer()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPEN_AI_API")

openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
supervisor_llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-flash-preview-05-20", temperature=0.0)

AGENT_LABEL_MAP = {
    "Agent 1": ["acquired_by", "merged_with", "subsidiary_of", "brand_of", "shareholder_of"],
    "Agent 2": ["competitor_of", "client_of", "collaboration"],
    "Agent 3": ["product_or_service_of", "regulated_by", "traded_on"]
}

AGENT_SPECIALIZATIONS = {
    "Agent 1": "Ownership or corporate-structure (control of the company itself)",
    "Agent 2": "Peer-to-peer business interactions (how two operating entities deal with each other)",
    "Agent 3": "Product / market-context relations (what the company offers and under what rules)"
}  

def supervisor_routing_prompt(sentence, head_entity, tail_entity):
    agent_info = "\n".join([f"{k}: {v}" for k, v in AGENT_SPECIALIZATIONS.items()])
    label_info = "\n".join([f"{agent}: {labels}" for agent, labels in AGENT_LABEL_MAP.items()])
    few_shot_examples = """
Examples:
Sentence: "It is owned by the Kongsberg Group and is part of its Kongsberg Defence & Aerospace division."
Entity 1: Kongsberg Spacetec AS
Entity 2: Kongsberg Defence & Aerospace
relation: subsidiary_of → Agent 1

Sentence: "International operations Time Life was also a financial backer for commercial TV broadcasting outside the United States, mostly in Middle and South America. With a joint venture between CBS and Goar Mestre they backed Proartel in Argentina, PROVENTEL in Venezuela (now VTV) and Panamericana Televisión in Peru."
Entity 1: Time Life Television
Entity 2: Goar Mestre
relation: collaboration → Agent 2

Sentence: "Indilinx's main product was its Barefoot series of flash controllers and their associated firmware for solid state drives. and regulated by the PRA and the Financial Conduct Authority in the United Kingdom."
Entity 1: Barefoot
Entity 2: Indilinx, Inc.
relation: product_or_service_of → Agent 3

Sentence: "On May 17, 2018, media outlets reported that the LocationSmart website allowed anyone to obtain the realtime location of any cell phone using any of the major U.S. wireless carriers (including AT&T, Verizon, T-Mobile, and Sprint), as well as some Canadian carriers, to within a few hundred yards, given only the phone number."
Entity 1: LocationSmart
Entity 2: T-Mobile
→ undefined
"""
    prompt = (
        f"You are a supervisor overseeing specialized relation classification agents.\n\n"
        f"Each agent specializes in a type of relationship between Entity 1 and Entity 2.\n\n"
        f"Agent expertise:\n{agent_info}\n\n"
        f"Each agent can handle only the following relation labels:\n{label_info}\n\n"
        f"Your task is to decide which agent should handle the classification for the sentence provided.\n\n"
        f"The sentence is:\n\"{sentence}\"\n\n"
        f"Entity 1: {head_entity}\n"
        f"Entity 2: {tail_entity}\n\n"
        f"If the sentence expresses a clear relationship between Entity 1 and Entity 2 that fits any agent’s domain, assign it to that agent.\n"
        f"ONLY respond with 'undefined' if the sentence does NOT express any meaningful relationship between Entity 1 and Entity 2.\n\n"
        f"{few_shot_examples}\n"
        f"Now respond with ONLY one of the following: 'Agent 1', 'Agent 2', 'Agent 3', or 'undefined'."
    )
    return prompt

def supervisor_feedback_prompt(sentence, agent_name, agent_label, correct_labels):
    return (
        f"You are supervising {agent_name} on the following sentence:\n"
        f"\"{sentence}\"\n\n"
        f"The agent proposed: \"{agent_label}\"\n"
        f"The set of possible correct labels for this agent: {', '.join(correct_labels)}\n\n"
        f"Is this answer appropriate (Y/N)? \n"
        f"If not, respond with constructive feedback, starting your reply with \"N: ...\" (no explanations before 'N:'), otherwise reply \"Y\"."
    )

def agent_prompt(agent_name, labels, sentence, specialization, head_entity, tail_entity, last_feedback=None, last_response=None):
    label_list = "\n".join(labels)

    few_shot_examples = {
        "Agent 1": '''
Examples:
Sentence: "The company was founded in 2008 by Ralph Firman, Sr. History Ralph Firman, Sr. was one of the founders of Van Diemen in 1973 which went on to become one of the leading formula car constructors in the world before being sold to Élan Motorsport Technologies in 1999."
Entity 1: Ralph Firman Racing
Entity 2: Élan Motorsport Technologies
→ acquired_by

Sentence: "It is owned by the Kongsberg Group and is part of its Kongsberg Defence & Aerospace division."
Entity 1: Kongsberg Spacetec AS
Entity 2: Kongsberg Defence & Aerospace
→ subsidiary_of
''',
        
        "Agent 2": '''
Examples:
Sentence: "International operations Time Life was also a financial backer for commercial TV broadcasting outside the United States, mostly in Middle and South America. With a joint venture between CBS and Goar Mestre they backed Proartel in Argentina, PROVENTEL in Venezuela (now VTV) and Panamericana Televisión in Peru."
Entity 1: Time Life Television
Entity 2: Goar Mestre
→ collaboration

Sentence: "During the 1950s and 1960s, INA supplied thousands of M950 and M953 submachine guns to the Brazilian Armed Forces, which remained in use from 1950 to 1972, mainly in the Brazilian Army, where it was the standard weapon of use during this period."
Entity 1: Brazilian Armed Forces
Entity 2: INA S/A Indústria Nacional de Armas
→ client_of
''',

        "Agent 3": '''
Examples:
Sentence: "Indilinx's main product was its Barefoot series of flash controllers and their associated firmware for solid state drives. and regulated by the PRA and the Financial Conduct Authority in the United Kingdom."
Entity 1: Barefoot
Entity 2: Indilinx, Inc.
→ product_or_service_of

Sentence: "Prior to the merger into KCG, GETCO traded in over 50 markets in North and South America, Europe and Asia, and was consistently among the top 5 participants by volume on many venues, including the CME, Eurex, NYSE Arca, NYSE Arca Options, BATS, Nasdaq, Nasdaq Options, Chi-X, BrokerTec, and eSpeed."
Entity 1: Global Electronic Trading Company
Entity 2: Nasdaq
→ traded_on
'''
    }

    prompt = (
        f"You are {agent_name}, a specialized expert in relation classification.\n"
        f"Your expertise: {specialization}\n\n"
        f"Given the sentence:\n\"{sentence}\"\n\n"
        f"Entity 1: {head_entity}\n"
        f"Entity 2: {tail_entity}\n\n"
        f"Classify the relationship between Entity 1 and Entity 2.\n"
        f"Respond with ONLY one label from the list below (copy exactly, no other words):\n"
        f"{label_list}\n\n"
        f"Here are some examples for guidance:\n"
        f"{few_shot_examples.get(agent_name, '')}\n\n"
    )

    if last_feedback and last_response:
        prompt += (
            f"Your previous answer was: '{last_response}'.\n"
            f"The supervisor provided the following feedback:\n{last_feedback}\n\n"
            f"Based explicitly on this feedback, select a DIFFERENT label from the above list. "
            f"Do NOT repeat your previous answer.\n"
        )

    return prompt

class RelationState(BaseModel):
    sentence: str
    head_entity: str
    tail_entity: str
    feedback: Optional[str] = None
    agent_response: Optional[str] = None
    last_agent_response: Optional[str] = None
    undefined_count: int = 0
    route: Optional[str] = None
    attempts: int = 0

def agent1_func(state: RelationState):
    state.attempts += 1
    prompt = agent_prompt(
        "Agent 1",
        AGENT_LABEL_MAP["Agent 1"],
        state.sentence,
        AGENT_SPECIALIZATIONS["Agent 1"],
        state.head_entity,
        state.tail_entity,
        state.feedback
    )
    agent_response = openai_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if agent_response == "undefined":
        state.undefined_count = state.undefined_count + 1
    else:
        state.undefined_count = 0 
    state.last_agent_response = state.agent_response
    state.agent_response = agent_response
    return state

def agent2_func(state: RelationState):
    state.attempts += 1
    prompt = agent_prompt(
        "Agent 2",
        AGENT_LABEL_MAP["Agent 2"],
        state.sentence,
        AGENT_SPECIALIZATIONS["Agent 2"],
        state.head_entity,
        state.tail_entity,
        state.feedback
    )
    agent_response = openai_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if agent_response == "undefined":
        state.undefined_count = state.undefined_count + 1
    else:
        state.undefined_count = 0
    state.last_agent_response = state.agent_response
    state.agent_response = agent_response
    return state

def agent3_func(state: RelationState):
    state.attempts += 1
    prompt = agent_prompt(
        "Agent 3",
        AGENT_LABEL_MAP["Agent 3"],
        state.sentence,
        AGENT_SPECIALIZATIONS["Agent 3"],
        state.head_entity,
        state.tail_entity,
        state.feedback
    )
    agent_response = openai_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if agent_response == "undefined":
        state.undefined_count = state.undefined_count + 1
    else:
        state.undefined_count = 0
    state.last_agent_response = state.agent_response
    state.agent_response = agent_response
    return state

def supervisor_router_func(state: RelationState):
    prompt = supervisor_routing_prompt(state.sentence, state.head_entity, state.tail_entity)
    response = supervisor_llm.invoke([HumanMessage(content=prompt)]).content.strip() 
    state.route = response
    return state

def supervisor_feedback_func(state):
    sentence = state.sentence
    agent = state.route
    agent_label = state.agent_response
    labels = AGENT_LABEL_MAP.get(agent, [])
    prompt = supervisor_feedback_prompt(sentence, agent, agent_label, labels)
    feedback = supervisor_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    state.feedback = feedback
    return state

def is_valid_feedback(feedback):
    return feedback.strip().startswith("Y")

graph = StateGraph(state_schema=RelationState)

def extract_route(text):
    if text is None:
        return "undefined"
    text = text.strip().lower()
    match = re.search(r'(agent\s*[123])|undefined', text, re.IGNORECASE)
    extracted = match.group(0).lower() if match else "undefined"
    return extracted

MAX_ATTEMPTS = 3
MAX_NO_RELATION = 2

def feedback_edge(state):
    print(f"[FEEDBACK_EDGE] feedback: {state.feedback}, attempts: {state.attempts}, undefined_count: {state.undefined_count}")
    if state.attempts >= MAX_ATTEMPTS:
        print(f"[FEEDBACK_EDGE] Exceeded max attempts ({MAX_ATTEMPTS}), stopping.")
        return END
    if state.undefined_count >= MAX_NO_RELATION:
        print(f"[FEEDBACK_EDGE] undefined returned too many times ({state.undefined_count}), stopping.")
        return END
    if is_valid_feedback(state.feedback):
        return END
    else:
        route = state.route
        if route == "Agent 1":
            return "agent1"
        elif route == "Agent 2":
            return "agent2"
        elif route == "Agent 3":
            return "agent3"
        else:
            return END

def routing_edge(state):
    raw_route = state.route if hasattr(state, 'route') and state.route else "undefined"
    route = extract_route(raw_route)
    if route == "agent 1":
        return "agent1"
    elif route == "agent 2":
        return "agent2"
    elif route == "agent 3":
        return "agent3"
    else:
        return END

graph.add_node("supervisor_router", supervisor_router_func)
graph.add_node("agent1", agent1_func)
graph.add_node("agent2", agent2_func)
graph.add_node("agent3", agent3_func)
graph.add_node("supervisor_feedback", supervisor_feedback_func)
graph.add_conditional_edges("supervisor_router", routing_edge)
graph.add_edge("agent1", "supervisor_feedback")
graph.add_edge("agent2", "supervisor_feedback")
graph.add_edge("agent3", "supervisor_feedback")
graph.add_conditional_edges("supervisor_feedback", feedback_edge)
graph.set_entry_point("supervisor_router")

def load_entity_story_rows_from_csv(csv_path: str):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row.get("sentence") or not row.get("entity_1") or not row.get("entity_2"):
                continue
            rows.append(
                {
                    "entity": row["entity_1"],
                    "target_entity": row["entity_2"],
                    "sentence": row["sentence"],
                    "label": row["relation"].strip().lower() if "relation" in row else "undefined"
                }
            )
    return rows

CSV_PATH = "./datasets/core_data/final_test.csv"
data = load_entity_story_rows_from_csv(CSV_PATH)

true_labels = []
predicted_labels = []

compiled_graph = graph.compile()

output_file = "predictions-multiAgent-CORE.csv"

fieldnames = ["sentence", "entity_1", "entity_2", "true_label", "predicted_label", "agent_route"]
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for item in data:
        state = RelationState(
            sentence=item["sentence"].strip(),
            head_entity=item["entity"].strip(),
            tail_entity=item["target_entity"].strip()
        )
    
        result = compiled_graph.invoke(state)
        if hasattr(result, "dict"):
            result = result.dict()
    
        if result.get("route", "undefined") == "undefined":
            final_label = "undefined"
        else:
            final_label = result.get("agent_response", "undefined")
    
        true_labels.append(item['label'].strip().lower())
        predicted_labels.append(final_label.strip().lower())

        row = {
            "sentence": item["sentence"],
            "entity_1": item["entity"],
            "entity_2": item["target_entity"],
            "true_label": item["label"],
            "predicted_label": final_label,
            "agent_route": result.get('route', 'undefined')
        }
        writer.writerow(row)
        f.flush()
 
acc = accuracy_score(true_labels, predicted_labels)
f1_macro = f1_score(true_labels, predicted_labels, average="macro")
f1_micro = f1_score(true_labels, predicted_labels, average="micro")

print(f"Accuracy     : {acc:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Micro     : {f1_micro:.4f}")