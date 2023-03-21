import pandas as pd

def save_csv(data, columns, file_name):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'../Experiments/{file_name}.csv')