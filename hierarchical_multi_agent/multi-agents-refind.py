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

openai_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0
)

supervisor_llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-flash-preview-05-20", temperature=0.0)

AGENT_LABEL_MAP = {
    "Agent 1": [
        "subsidiary_of", "shares_of", "acquired_by", "operations_in", "headquartered_in", "formed_in", "agreement_with"
    ],
    "Agent 2": [
        "formed_on", "acquired_on", "revenue_of", "profit_of", "loss_of", "cost_of", 
    ],
    "Agent 3": [
        "member_of", "employee_of", "attended", "founder_of", "title"
    ]
}

AGENT_SPECIALIZATIONS = {
    "Agent 1": "Relationships between organizations or geopolitical entities.",
    "Agent 2": "Relationships involving dates or monetary/numerical values.",
    "Agent 3": "Relationships between persons, organizations, titles, or universities."
}  

def supervisor_routing_prompt(sentence, head_entity, tail_entity):
    agent_info = "\n".join([f"{k}: {v}" for k, v in AGENT_SPECIALIZATIONS.items()])
    label_info = "\n".join([f"{agent}: {labels}" for agent, labels in AGENT_LABEL_MAP.items()])
    few_shot_examples = """
Examples:
Sentence: "In September 2014 , as part of the removal of anti - dilution , price reset and change of control provisions in various securities that had caused those securities to be classified as derivative liabilities , CorMedix Inc. entered into a Consent and Exchange Agreement with Manchester, pursuant to which Manchester had a right of 60 % participation in equity financings undertaken by CorMedix Inc."
Entity 1: CorMedix Inc.
Entity 2: Manchester
Label: agreement_with → Agent 1

Sentence: "Illinois EMCASCO was formed in Illinois in 1976 (and was re-domesticated to Iowa in 2001), Dakota Fire was formed in North Dakota in 1957 and EMCASCO was formed in Iowa in 1958, all for the purpose of writing property and casualty insurance."
Entity 1: EMCASCO
Entity 2: 1976
Label: formed_on → Agent 2

Sentence: "Dr. Smith also served as a member of the Industrial Associates of the School of Earth Sciences at Stanford University for several years."
Entity 1: Smith
Entity 2: the Industrial Associates of the School of Earth Sciences
Label: member_of → Agent 3

Sentence: "According to data received from Smith Travel Research and compiled by us in order to analyze MARCUS CORP fiscal year results, comparable upper upscale hotels throughout the United States experienced an increase in RevPAR of 6.9 % during MARCUS CORP fiscal 2015."
Entity 1: MARCUS CORP
Entity 2: fiscal year
→ no_relation
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
        f"ONLY respond with 'no_relation' if the sentence does NOT express any meaningful relationship between Entity 1 and Entity 2.\n\n"
        f"{few_shot_examples}\n"
        f"Now respond with ONLY one of the following: 'Agent 1', 'Agent 2', 'Agent 3', or 'no_relation'."
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
Sentence: "In September 2014 , as part of the removal of anti - dilution , price reset and change of control provisions in various securities that had caused those securities to be classified as derivative liabilities , CorMedix Inc. entered into a Consent and Exchange Agreement with Manchester, pursuant to which Manchester had a right of 60 % participation in equity financings undertaken by CorMedix Inc."
Entity 1: CorMedix Inc.
Entity 2: Manchester
→ agreement_with

Sentence: "In the United States, NUVASIVE INC sell NUVASIVE INC products through a combination of exclusive independent sales agents and directly-employed sales personnel."
Entity 1: NUVASIVE INC
Entity 2: the United States
→ operations_in
''',

        "Agent 2": '''
Examples:
Sentence: "Illinois EMCASCO was formed in Illinois in 1976 (and was re-domesticated to Iowa in 2001), Dakota Fire was formed in North Dakota in 1957 and EMCASCO was formed in Iowa in 1958, all for the purpose of writing property and casualty insurance."
Entity 1: EMCASCO
Entity 2: 1976
→ formed_on

Sentence: "DELTA APPAREL , INC operating profit, without adjusting for Junkfood as mentioned above, was $7.5 million, or 7.2% of sales."
Entity 1: DELTA APPAREL , INC
Entity 2: $ 7.5 million
→ profit_of
''',

        "Agent 3": '''
Examples:
Sentence: "Dr. Smith also served as a member of the Industrial Associates of the School of Earth Sciences at Stanford University for several years."
Entity 1: Smith
Entity 2: the Industrial Associates of the School of Earth Sciences
→ member_of

Sentence: "Mr. Schriesheim also served as a director of Dobson Communications Corp. from 2004 to 2007, a director of Lawson Software from 2006 to 2011, a director and Co-Chairman of MSC Software Corporation from 2007 to 2009 and a director of Georgia Gulf Corporation from 2009 to 2010."
Entity 1: Schriesheim
Entity 2: Lawson Software
→ employee_of
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
    no_relation_count: int = 0
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
    if agent_response == "no_relation":
        state.no_relation_count = state.no_relation_count + 1
    else:
        state.no_relation_count = 0
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
    if agent_response == "no_relation":
        state.no_relation_count = state.no_relation_count + 1
    else:
        state.no_relation_count = 0
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
    if agent_response == "no_relation":
        state.no_relation_count = state.no_relation_count + 1
    else:
        state.no_relation_count = 0
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
        return "no_relation"
    text = text.strip().lower()
    match = re.search(r'(agent\s*[123])|no_relation', text, re.IGNORECASE)
    extracted = match.group(0).lower() if match else "no_relation"
    return extracted

MAX_ATTEMPTS = 3
MAX_NO_RELATION = 2

def feedback_edge(state):
    print(f"[FEEDBACK_EDGE] feedback: {state.feedback}, attempts: {state.attempts}, no_relation_count: {state.no_relation_count}")
    if state.attempts >= MAX_ATTEMPTS:
        print(f"[FEEDBACK_EDGE] Exceeded max attempts ({MAX_ATTEMPTS}), stopping.")
        return END
    if state.no_relation_count >= MAX_NO_RELATION:
        print(f"[FEEDBACK_EDGE] no_relation returned too many times ({state.no_relation_count}), stopping.")
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
    raw_route = state.route if hasattr(state, 'route') and state.route else "no_relation"
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
            if not row.get("sentence") or not row.get("head_entity_text") or not row.get("tail_entity_text"):
                continue
            rows.append(
                {
                    "entity": row["head_entity_text"],
                    "target_entity": row["tail_entity_text"],
                    "sentence": row["sentence"],
                    "label": row["relation"].strip().lower() if "relation" in row else "no_relation"
                }
            )
    return rows

CSV_PATH = "./datasets/refind_data/test_refined.csv"
data = load_entity_story_rows_from_csv(CSV_PATH)

true_labels = []
predicted_labels = []

compiled_graph = graph.compile()

output_file = "predictions-multiAgent-refind.csv"

fieldnames = ["sentence", "head_entity_text", "tail_entity_text", "true_label", "predicted_label", "agent_route"]
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
    
        if result.get("route", "no_relation") == "no_relation":
            final_label = "no_relation"
        else:
            final_label = result.get("agent_response", "no_relation")
    
        true_labels.append(item['label'].strip().lower())
        predicted_labels.append(final_label.strip().lower())

        row = {
            "sentence": item["sentence"],
            "head_entity_text": item["entity"],
            "tail_entity_text": item["target_entity"],
            "true_label": item["label"],
            "predicted_label": final_label,
            "agent_route": result.get('route', 'no_relation')
        }
        writer.writerow(row)
        f.flush()
 
acc = accuracy_score(true_labels, predicted_labels)
f1_macro = f1_score(true_labels, predicted_labels, average="macro")
f1_micro = f1_score(true_labels, predicted_labels, average="micro")

print(f"Accuracy     : {acc:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Micro     : {f1_micro:.4f}")