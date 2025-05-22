import pandas as pd

def filter_relations(input_csv='test_semeval.csv', output_csv='test_semeval_filtered.csv', max_per_relation=25, random_state=42):
    """
    Filter relations so that no relation_name appears more than max_per_relation times.
    Rows per relation_name are randomly sampled if needed.

    Parameters:
        input_csv (str): Path to input CSV file.
        output_csv (str): Path to the filtered output CSV file.
        max_per_relation (int): Maximum rows to keep per unique relation_name.
        random_state (int): Random state for reproducibility in sampling.
    """
    df = pd.read_csv(input_csv)
    filtered = (
        df.groupby("relation_name", group_keys=False)
        .apply(lambda x: x.sample(n=min(max_per_relation, len(x)), random_state=random_state))
    )
    filtered.to_csv(output_csv, index=False)

# Example usage:
if __name__ == "__main__":
    filter_relations()  # Uses defaults from above