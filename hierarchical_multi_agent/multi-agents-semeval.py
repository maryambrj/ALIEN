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
    "Agent 1": ["Cause-Effect (e1, e2)", "Cause-Effect (e2, e1)",
                "Product-Producer (e1, e2)", "Product-Producer (e2, e1)",
                "Instrument-Agency (e1, e2)", "Instrument-Agency (e2, e1)"],

    "Agent 2": ["Component-Whole (e1, e2)", "Component-Whole (e2, e1)",
                "Content-Container (e1, e2)", "Content-Container (e2, e1)",
                "Member-Collection (e1, e2)", "Member-Collection (e2, e1)"],

    "Agent 3": ["Entity-Destination (e1, e2)", "Entity-Destination (e2, e1)",
                "Entity-Origin (e1, e2)", "Entity-Origin (e2, e1)",
                "Message-Topic (e1, e2)", "Message-Topic (e2, e1)"]
}

AGENT_SPECIALIZATIONS = {
    "Agent 1": "Causal, functional, or productive relationships (actions, tools, producers).",
    "Agent 2": "Structural or inclusion-based relationships (parts, members, containers).",
    "Agent 3": "Movement, origin, or informational associations (where things go, come from, or what they're about)."
} 

def supervisor_routing_prompt(sentence, head_entity, tail_entity):
    agent_info = "\n".join([f"{k}: {v}" for k, v in AGENT_SPECIALIZATIONS.items()])
    label_info = "\n".join([f"{agent}: {labels}" for agent, labels in AGENT_LABEL_MAP.items()])
    few_shot_examples = """
Examples:
Sentence: "The team stapled the plastic along the joists with heavy duty staple guns to hold it in place."
Entity 1: team
Entity 2: guns
relation: Instrument-Agency (e2, e1) → Agent 1

Sentence: "The barbels of the exposed catfish curled within 2 hours in heptachlor."
Entity 1: barbels
Entity 2: catfish
relation: Component-Whole (e1, e2) → Agent 2

Sentence: "The ship is arriving into the port now."
Entity 1: ship
Entity 2: port
relation: Entity-Destination (e1, e2) → Agent 3

Sentence: "The opening angle of the Cherenkov cone measures the velocity with a resolution of 0.2%."
Entity 1: angle
Entity 2: resolution
→ other
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
        f"If the sentence expresses a clear relationship between Entity 1 and Entity 2 (in either direction) that fits any agent’s domain, assign it to that agent.\n"
        f"ONLY respond with 'other' if the sentence does NOT express any meaningful relationship between Entity 1 and Entity 2.\n\n"
        f"{few_shot_examples}\n"
        f"Now respond with ONLY one of the following: 'Agent 1', 'Agent 2', 'Agent 3', or 'other'."
    )
    return prompt

def supervisor_feedback_prompt(sentence, agent_name, agent_label, correct_labels):
    return (
        f"You are supervising {agent_name} on the following sentence:\n"
        f"\"{sentence}\"\n\n"
        f"The agent proposed: \"{agent_label}\"\n"
        f"The set of possible correct labels for this agent (with direction): {', '.join(correct_labels)}\n\n"
        f"The label must match both the relation type and direction (e1, e2 vs e2, e1).\n"
        f"Is this answer appropriate (Y/N)? \n"
        f"If not, respond with constructive feedback, starting your reply with \"N: ...\" (no explanations before 'N:'), otherwise reply \"Y\"."
    )

def agent_prompt(agent_name, labels, sentence, specialization, head_entity, tail_entity, last_feedback=None, last_response=None):
    label_list = "\n".join(labels)

    few_shot_examples = {
        "Agent 1": '''
Examples:
Sentence: "The team stapled the plastic along the joists with heavy duty staple guns to hold it in place."
Entity 1: team
Entity 2: guns
→ Instrument-Agency (e2, e1)

Sentence: "Sadness leads to dissatisfaction with the job."
Entity 1: Sadness
Entity 2: dissatisfaction
→ Cause-Effect (e1, e2)
''',

        "Agent 2": '''
Examples:
Sentence: "The barbels of the exposed catfish curled within 2 hours in heptachlor."
Entity 1: barbels
Entity 2: catfish
→ Component-Whole (e1, e2)

Sentence: "His literary debut occured in 1919, when he joined Esenin's small circle of "literary hooligans," the Imaginists."
Entity 1: circle
Entity 2: hooligans
→ Member-Collection (e2, e1)
''',

        "Agent 3": '''
Examples:
Sentence: "The ship is arriving into the port now."
Entity 1: ship
Entity 2: port
→ Entity-Destination (e1, e2)

Sentence: "This book has been the topic of considerable heated debate, but it is not the intention of this reviewer to enter into this public argument."
Entity 1: book
Entity 2: debate
→ Message-Topic (e2, e1)
'''
    }

    prompt = (
        f"You are {agent_name}, a specialized expert in relation classification.\n"
        f"Your expertise: {specialization}\n\n"
        f"Given the sentence:\n\"{sentence}\"\n\n"
        f"Entity 1: {head_entity}\n"
        f"Entity 2: {tail_entity}\n\n"
        f"Classify the relationship between Entity 1 and Entity 2 including the direction.\n"
        f"The direction (e1, e2) means the relation flows from Entity 1 to Entity 2. (e2, e1) means from Entity 2 to Entity 1.\n"
        f"Respond with ONLY one label from the list below (copy exactly, no other words, but include directionality):\n"
        f"{label_list}\n\n"
        f"Here are some examples for guidance:\n"
        f"{few_shot_examples.get(agent_name, '')}\n\n"
    )

    if last_feedback and last_response:
        prompt += (
            f"Your previous answer was: '{last_response}'.\n"
            f"The supervisor provided the following feedback:\n{last_feedback}\n\n"
            f"Based explicitly on this feedback, select a DIFFERENT label from the above list. Your answer must match the correct relation and direction."
            f"Do NOT repeat your previous answer. Pick a new label that better reflects the sentence, entity pair, and correct direction.\n"
        )

    return prompt

class RelationState(BaseModel):
    sentence: str
    head_entity: str
    tail_entity: str
    feedback: Optional[str] = None
    agent_response: Optional[str] = None
    last_agent_response: Optional[str] = None
    other_count: int = 0
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
    if not re.match(r".+\((e1, e2|e2, e1)\)", agent_response, re.IGNORECASE):
        print(f"Invalid label format from agent: {agent_response}")

    if agent_response == "other":
        state.other_count = state.other_count + 1
    else:
        state.other_count = 0
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
    if not re.match(r".+\((e1, e2|e2, e1)\)", agent_response, re.IGNORECASE):
        print(f"Invalid label format from agent: {agent_response}")
        
    if agent_response == "other":
        state.other_count = state.other_count + 1
    else:
        state.other_count = 0
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
    if not re.match(r".+\((e1, e2|e2, e1)\)", agent_response, re.IGNORECASE):
        print(f"Invalid label format from agent: {agent_response}")
        
    if agent_response == "other":
        state.other_count = state.other_count + 1
    else:
        state.other_count = 0
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
        return "other"
    text = text.strip().lower()
    match = re.search(r'(agent\s*[123])|other', text, re.IGNORECASE)
    extracted = match.group(0).lower() if match else "other"
    return extracted

MAX_ATTEMPTS = 3
MAX_NO_RELATION = 2

def feedback_edge(state):
    print(f"[FEEDBACK_EDGE] feedback: {state.feedback}, attempts: {state.attempts}, other_count: {state.other_count}")
    if state.attempts >= MAX_ATTEMPTS:
        print(f"[FEEDBACK_EDGE] Exceeded max attempts ({MAX_ATTEMPTS}), stopping.")
        return END
    if state.other_count >= MAX_NO_RELATION:
        print(f"[FEEDBACK_EDGE] other returned too many times ({state.other_count}), stopping.")
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
    raw_route = state.route if hasattr(state, 'route') and state.route else "other"
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
                    "label": row["relation"].strip().lower() if "relation" in row else "other"
                }
            )
    return rows

CSV_PATH = "./datasets/semeval_2010_task8/final_test.csv"
data = load_entity_story_rows_from_csv(CSV_PATH)

true_labels = []
predicted_labels = []

compiled_graph = graph.compile()

output_file = "predictions-multiAgent-Semeval.csv"

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
    
        if result.get("route", "other") == "other":
            final_label = "other"
        else:
            final_label = result.get("agent_response", "other")
    
        true_labels.append(item['label'].strip().lower())
        predicted_labels.append(final_label.strip().lower())

        row = {
            "sentence": item["sentence"],
            "entity_1": item["entity"],
            "entity_2": item["target_entity"],
            "true_label": item["label"],
            "predicted_label": final_label,
            "agent_route": result.get('route', 'other')
        }
        writer.writerow(row)
        f.flush() 
 
acc = accuracy_score(true_labels, predicted_labels)
f1_macro = f1_score(true_labels, predicted_labels, average="macro")
f1_micro = f1_score(true_labels, predicted_labels, average="micro")

print(f"Accuracy     : {acc:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Micro     : {f1_micro:.4f}")