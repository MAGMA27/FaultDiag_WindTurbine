import pandas as pd

data = pd.read_parquet(path = r'D:\Dev\FaultDiagnose\data\processed\farm_A\0.parquet')

print(data.head())