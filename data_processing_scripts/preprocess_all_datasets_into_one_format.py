import random
import pandas as pd

random.seed(0)


def reformat_core_data(df):
    df["sentence"] = df["context"].apply(lambda x: " ".join(x).strip())
    df.rename(
        columns={
            "e1_name": "entity_1",
            "e2_name": "entity_2",
        },
        inplace=True
    )
    mask = df["invert_relation"] == 1
    entity_1 = df.loc[mask, "entity_2"]
    entity_2 = df.loc[mask, "entity_1"]
    df.loc[mask, "entity_1"] = entity_1
    df.loc[mask, "entity_2"] = entity_2
    df.drop(
        columns=[
            "context",
            "invert_relation",
            "e1_start",
            "e1_end",
            "e2_start",
            "e2_end"
        ],
        inplace=True
    )
    return df


def reformat_semeval_data(df):
    df.drop(
        columns=[
            "sentence",
            "label", "split"
        ],
        inplace=True
    )
    df.rename(
        columns={
            "entity1": "entity_1",
            "entity2": "entity_2",
            "sentence_without_tags": "sentence"
        },
        inplace=True
    )
    return df


def reformat_refind_data(df):
    df.drop(
        columns=[
            "id", "docid",
            "head_entity_char_idxs", "tail_entity_char_idxs",
            "label", "relation_group",
            "e1_type", "e2_type", "sentence_with_entity_tags"
        ],
        inplace=True
    )
    df.rename(
        columns={
            "head_entity_text": "entity_1",
            "tail_entity_text": "entity_2"
        },
        inplace=True
    )
    return df


def process_core_dataset():
    df_train = reformat_core_data(
        pd.read_json("./../datasets/core_data/original_dataset/train.json")
    )
    df_test = reformat_core_data(
        pd.read_json("./../datasets/core_data/original_dataset/test.json")
    )
    df_train.to_csv("./../datasets/core_data/train.csv", index=False)
    df_test.to_csv("./../datasets/core_data/test.csv", index=False)


def process_semeval_dataset():
    df_train = pd.read_csv(
        "./../datasets/semeval_2008_task8/original_dataset/train.csv")
    df_test = pd.read_csv(
        "./../datasets/semeval_2008_task8/original_dataset/test.csv")
    df_train = reformat_semeval_data(df_train)
    df_test = reformat_semeval_data(df_test)
    df_train.to_csv(
        "./../datasets/semeval_2008_task8/train.csv", index=False)
    df_test.to_csv(
        "./../datasets/semeval_2008_task8/test.csv", index=False)


def process_refind_dataset():
    df_train = pd.read_csv(
        "./../datasets/refind_data/original_dataset/train.csv")
    df_test = pd.read_csv(
        "./../datasets/refind_data/original_dataset/test.csv")
    df_train = reformat_refind_data(df_train)
    df_test = reformat_refind_data(df_test)
    df_train.to_csv("./../datasets/refind_data/train.csv", index=False)
    df_test.to_csv("./../datasets/refind_data/test.csv", index=False)


if __name__ == "__main__":
    process_core_dataset()
    process_semeval_dataset()
    process_refind_dataset()
