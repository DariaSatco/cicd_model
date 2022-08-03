import pandas as pd
import csv
import yaml

from sklearn.model_selection import train_test_split

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

    raw_data_pth = params['paths']['raw_data']
    prep_data_pth = params['paths']['preprocessed_data']
    test_data_pth = params['paths']['test_data']

    clean_data = remove_whitespc(raw_data_pth)
    
    # keep piece of data for unit tests to evaluate updated model
    train_data, test_data = train_test_split(clean_data, 
                                             test_size=0.1, 
                                             stratify=clean_data['salary'],
                                             random_state=10)

    train_data.to_csv(prep_data_pth, index=False)
    test_data.to_csv(test_data_pth, index=False)
