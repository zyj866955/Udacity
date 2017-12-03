import pandas as pd
# datetime = pd.date_range(start='20171101', end='20171110')
#
# print(datetime[1])

df = pd.DataFrame([{'col1':'a', 'col2':1}, {'col1':'b', 'col2':2}, {'col1':'c', 'col2':3}])
# df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
# print(df)
# df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True,False] * 2})
# print(df_mask)
# print(df.where(df_mask))

# print(df)
# print(df['col1'] == 'a')
# 单条件
# print (df.loc[df['col1'] == 'a'])
# print("*************")
# print(df.loc[0])
# print (df.loc[df['col1'] != 'a'])
# print (df.loc[df['col2'] > 2])
# print(df['col1'].isin(['a','b']))
# df.where
# print (df.loc[df['col1'].isin(['a', 'b'])])

# 多条件
# print (df.loc[df['col1'] == 'c'].loc[df['col2'] > 1])


def matrix(shape):
    temp = [[ 0 for i in range(4) ] for j in range(4)]
    for i in range(shape):
        for j in range(shape):
            if i == j:
                temp[i][j] = 1
    return temp
I = matrix(4)

a = [1,2,3,4]
print()