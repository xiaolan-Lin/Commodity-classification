import pandas as pd


def read_file():
    """
    导入数据
    """
    commodity = pd.read_csv(r"D:\PycharmProjects\Commodity-classification\data\train.tsv", sep='\t')
    # 查看分类种类
    commodity['TYPE'].unique()
    print(len(commodity['TYPE'].value_counts()))

    return commodity


if __name__ == '__main__':
    read_file()






