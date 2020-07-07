import pandas as pd


def read_file():
    commodity = pd.read_csv("/home/master/pythonProject/Commodity-classfication/data/train.tsv", sep='\t')
    return commodity


if __name__ == '__main__':
    read_file()


commodity = pd.read_csv("/home/master/pythonProject/Commodity-classfication/data/train.tsv", sep='\t')

# 查看分类种类
commodity['TYPE'].unique()
print(len(commodity['TYPE'].value_counts()))


