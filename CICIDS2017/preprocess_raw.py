import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA


# 读取资料集
dataframe1 = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
dataframe2 = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", low_memory=False)
dataframe7 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv", low_memory=False)

# 取出4种label的资料
df1 = dataframe2.loc[dataframe2[' Label'] == 'BENIGN']  # [127537 rows x 79 columns]
df2 = dataframe1.loc[dataframe1[' Label'] == 'DDoS']  # [128027 rows x 79 columns]
df3 = dataframe2.loc[dataframe2[' Label'] == 'PortScan']  # [158930 rows x 79 columns]
df3 = df3[0:130000]
df4 = dataframe7.loc[dataframe7[' Label'] == 'DoS Hulk']  # [231073 rows x 79 columns]
df4 = df4[0:130000]

# 合并成一个资料集
dataframe = pd.concat([df1, df2, df3, df4])
dataframe = shuffle(dataframe, random_state=8)  # 合并后打乱

# 补齐缺失值: Flow Bytes/s
dataframe.loc[:, 'Flow Bytes/s'] = dataframe.loc[:, 'Flow Bytes/s'].replace(np.nan, 0, regex=True)

# 修改列名
dataframe.columns = ['Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
                     'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max',
                     'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean', 'Fwd_Packet_Length_Std',
                     'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
                     'Bwd_Packet_Length_Std', 'Flow_Bytes', 'Flow_Packets', 'Flow_IAT_Mean', 'Flow_IAT_Std',
                     'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max',
                     'Fwd_IAT_Min', 'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min',
                     'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags', 'Fwd_Header_Length',
                     'Bwd_Header_Length', 'Fwd_Packets', 'Bwd_Packets', 'Min_Packet_Length', 'Max_Packet_Length',
                     'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count',
                     'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count',
                     'CWE_Flag_Count', 'ECE_Flag_Count', 'Down', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size',
                     'Avg_Bwd_Segment_Size', 'Fwd_Header_Length.1', 'Fwd_Avg_Bytes', 'Fwd_Avg_Packets',
                     'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes', 'Bwd_Avg_Packets', 'Bwd_Avg_Bulk_Rate',
                     'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes', 'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes',
                     'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                     'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max',
                     'Idle_Min', 'Label']
# 重置索引
dataframe = dataframe.reset_index(drop=True)
print('初始Data:')
print(dataframe)

# 取出数据和标签
FL_data = dataframe.drop(['Label'],axis=1)
FL_label = dataframe['Label']
FL_data.replace(np.inf, 0, inplace=True)  # 去除极大值
print('只保留數據:')
print(FL_data)
print(FL_label)
FL_data.to_csv('CIC_data.csv', index=False)
changelabel = {'BENIGN' : '0','DDoS' : '1', 'DoS Hulk' : '1','PortScan' : '1'}
FL_label = FL_label.map(changelabel)
FL_label.to_csv('CIC_label.csv', index=False, header=False)

print("Label各元素及個數: ")
print(FL_label.value_counts(), '\n')

