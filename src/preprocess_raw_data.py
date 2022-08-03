import pandas as pd
import csv
import yaml


def remove_whitespc(csv_path: str) -> pd.DataFrame:
    """
    Reads csv file and strips cell values to remove
    redundant whitespaces
    """
    aList = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            aList.append([x.strip() for x in row])

    clean_df = pd.DataFrame(aList[1:-1], columns=aList[0])

    return clean_df


if __name__ == "__main__":

    with open('model_config.yaml') as f:
        params = yaml.safe_load(f)
    print(params)

    raw_data_pth = params['paths']['raw_data']
    prep_data_pth = params['paths']['preprocessed_data']

    clean_data = remove_whitespc(raw_data_pth)
    clean_data.to_csv(prep_data_pth, index=False)
