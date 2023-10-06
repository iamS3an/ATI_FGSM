import pandas as pd
import numpy as np


CIC_data = pd.read_csv("CIC_data.csv", low_memory=False)
CIC_data.columns = range(CIC_data.shape[1])

max_list = []
min_list = []

max_data = CIC_data.max()
min_data = CIC_data.min()

max_list.append(max_data.tolist())
min_list.append(min_data.tolist())

max_df = pd.DataFrame(np.array(max_list), dtype='float64')
min_df = pd.DataFrame(np.array(min_list), dtype='float64')

print(max_df)
print(min_df)
print(CIC_data)

denominator = max_df - min_df
print(denominator)
new_data = CIC_data.subtract(min_df.iloc[0])
print(new_data)
new_data = new_data.div(denominator.iloc[0])
print(new_data)
new_data.fillna(0, inplace=True)
new_data.replace("", "0", inplace=True)  # 補空值
new_data.to_csv("CIC_normalize.csv", index=False, header=False)