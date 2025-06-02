from extraction_client import ExtractionClient
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
import os
import asyncio
import argparse
import random
from tqdm import tqdm
from sklearn.metrics import f1_score
random.seed(42)  # For reproducibility


load_dotenv("./../.env")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-shot relation forger agent script")
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Model to use for inference"
    )
    parser.add_argument(
        "--input_df_path", type=str,
        required=True, help="Input CSV file path for prediction"
    )
    parser.add_argument(
        "--input_train_df_path", type=str, required=True,
        help="Input CSV file path for training data"
    )
    parser.add_argument(
        "--output_file_path", type=str, required=True,
        help="Output df path"
    )
    parser.add_argument(
        "--n_shots", type=int, default=0,
        help="Number of few-shot examples to use"
    )
    return parser.parse_args()


def validate_df_has_required_columns(df):
    required_columns = ['sentence', 'entity_1', 'entity_2', 'relation']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column.")
    return True


def prepare_few_shot_prompt(input_df, n_shots=0):
    few_shot_prompt = ""
    if n_shots > 0:
        input_df = input_df.sample(n=n_shots, random_state=42)
        for _, row in input_df.iterrows():
            few_shot_prompt += (
                f"Context: {row['sentence']}\n"
                f"Entity 1: {row['entity_1']}\n"
                f"Entity 2: {row['entity_2']}\n"
                f"Relation : {row['relation']}\n\n"
            )
    return few_shot_prompt


async def create_prompt_and_call_api(
    input_datapoint_dict, few_shot_prompt,
    client, data_model, all_relations
):
    input_prompt = f"""
    You are a helpful assistant that extracts relationships between entities in a sentence. Given a context and entities identify the relationship from the provided list of labels.
    "List of Labels: {all_relations}\n"""

    if len(few_shot_prompt) > 0:
        input_prompt += f"\n\nSome examples are :\n{few_shot_prompt}\n\n"

    input_prompt += (
        f"Context: {input_datapoint_dict['sentence']}\n"
        f"Entity 1: {input_datapoint_dict['entity_1']}\n"
        f"Entity 2: {input_datapoint_dict['entity_2']}\n"
    )
    try:
        response = await client.extract(
            prompt=input_prompt,
            schema=data_model,
            system_prompt=""
        )
    except Exception as e:
        print(e)
        response = {
            # Fallback to a random relation
            "relationship": random.choice(all_relations)
        }
    return response


async def run_prediction(
    input_df, few_shot_prompt, client, data_model, all_relations
):
    # iterate asynchronouly over the test dataframe
    tasks = [
        create_prompt_and_call_api(
            input_datapoint_dict=row.to_dict(),
            few_shot_prompt=few_shot_prompt,
            client=client,
            data_model=data_model,
            all_relations=all_relations
        )
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Processing rows")
    ]
    all_api_outputs = await asyncio.gather(*tasks)
    all_api_outputs = [
        api_output["relationship"]
        for api_output in all_api_outputs
    ]
    return all_api_outputs


if __name__ == "__main__":
    args = parse_args()

    if "gpt" in args.model.lower():
        # For OpenAI GPT-4o
        client = ExtractionClient(
            provider="openai",
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name=args.model,
        )
    else:
        # For Gemini 2 Flash
        client = ExtractionClient(
            provider="gemini",
            api_key=os.environ.get('GENAI_API_KEY'),
            model_name=args.model,
            max_retries=20
        )
    # load dataframes
    train_df = pd.read_csv(args.input_train_df_path)
    test_df = pd.read_csv(args.input_df_path)

    # validate input train and test dataframes
    if not validate_df_has_required_columns(train_df) or not validate_df_has_required_columns(test_df):
        raise ValueError(
            "Input DataFrame must contain 'sentence', 'entity_1',"
            " 'entity_2', and 'relation' columns."
        )

    few_shot_prompt = prepare_few_shot_prompt(train_df, args.n_shots)

    all_relations = train_df.relation.unique().tolist()

    class ClassificationOutput(BaseModel):
        relationship: Literal[*all_relations] = Field(
            description="Predicted relationship between the entities."
        )

    all_api_outputs = asyncio.run(run_prediction(
        input_df=test_df,
        few_shot_prompt=few_shot_prompt,
        client=client,
        data_model=ClassificationOutput,
        all_relations=all_relations
    ))
    # now map api_outputs to the test dataframe
    test_df["predicted_relation"] = all_api_outputs
    print(
        f"F1 Score: {f1_score(test_df['relation'], test_df['predicted_relation'], average='micro'):.2f}")
    test_df.to_csv(args.output_file_path, index=False)
