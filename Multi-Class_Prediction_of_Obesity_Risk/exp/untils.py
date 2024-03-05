import math
from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder



    

class DataPlot:
    def __init__(self, df:pd.DataFrame,scale=1.0):
        self.df = df
        self.figsize = (6.5*scale, 5*scale)
    
    def plot_category(self, column, line=None,scale=None):
        if scale is not None:
            plt.figure(figsize=(6.5*scale, 5*scale))
        else:
            plt.figure(figsize=self.figsize)
             
        

    def plot_compare_category_number(
            self,
            column1,
            column2,
            bins=10, 
            exclude=[],
            scale=None,
            line=None,
        ):
        if scale is not None:
            plt.figure(figsize=(6.5*scale, 5*scale))
        else:
            plt.figure(figsize=self.figsize)
        #column1がカテゴリ変数かを判定
        if self.df[column1].dtype == "object":
            raise ValueError("column1 must be category variable")
        #column2が数値変数かを判定
        if self.df[column2].dtype != "float64" and self.df[column2].dtype != "int64":
            raise ValueError("column2 must be numerical variable")
        
        for uniq in self.df[column1].unique():
            if uniq is np.nan:
                continue
            if math.isnan(uniq):
                continue
            if uniq in exclude:
                continue

            # 表示メイン
            # sns.distplot(self.df[self.df[column1]==uniq][column2], kde=True, rug=False, bins=bins, label=uniq)
            sns.histplot(data=self.df[self.df[column1]==uniq], x=column2, kde=True, bins=bins, label=uniq)
        if line is not None:
            if type(line) == list:
                for l in line:
                    plt.axvline(x=l, color="red", linestyle="--")
            else:
                plt.axvline(x=line, color="red", linestyle="--")

        plt.legend()
        plt.tight_layout()
        plt.show()

class DataCTR(DataPlot):
    def __init__(self, df: pd.DataFrame, scale=1):
        super().__init__(df, scale)
        self.df = df.copy()
    
    def dummy_category(self,
                       label_column=[],
                       one_hot_column=[],
                       inplace=True):
        if inplace:
            self.label_category(label_column, inplace=True)
            self.one_hot_category(one_hot_column, inplace=True)
            return self.df
        else:
            df = self.df.copy()
            df = self.label_category(label_column, inplace=False)
            df = self.one_hot_category(one_hot_column, inplace=False)
            return df


    def label_category(self, label_column, inplace=True):
        if inplace:
            df = self.df
        else:
            df = self.df.copy()
        le = LabelEncoder()
        for col in label_column:
            df[col] = le.fit_transform(df[col])
        return df
    
    def one_hot_category(self, one_hot_column, inplace=True,drop_first=True,**kwargs):
        if inplace:
            self.df = pd.get_dummies(self.df, columns=one_hot_column, drop_first=drop_first, **kwargs)
            return self.df
        else:
            return pd.get_dummies(self.df, columns=one_hot_column, drop_first=drop_first, **kwargs)
        

    def ordinal_category(self, column:str, order:dict,reverse=False, inplace=True):
        if inplace:
            df = self.df
        else:
            df = self.df.copy()
        
        #reverseがTrueの場合、orderを逆にする
        if reverse:
            order = {v:k for k,v in order.items()}

        #orderをもとに変換
        df[column] = df[column].replace(order)
        return df