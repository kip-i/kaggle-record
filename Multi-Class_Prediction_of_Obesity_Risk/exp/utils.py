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
    
        

    def plot_compare_category_number(
        self,
        column1,
        column2,
        bins=10, 
        exclude=[],
        scale=None,
        line=None,
    ):
        """
        カテゴリ変数と数値変数の比較をプロットします。

        Parameters:
            column1 (str): カテゴリ変数の列名
            column2 (str): 数値変数の列名
            bins (int, optional): ビンの数 (デフォルトは10)
            exclude (list, optional): 除外するカテゴリのリスト (デフォルトは空リスト)
            scale (float, optional): プロットのスケール (デフォルトはNone)
            line (float or list, optional): 追加の縦線の位置 (デフォルトはNone)

        Raises:
            ValueError: column1がカテゴリ変数でない場合、column2が数値変数でない場合

        Returns:
            None
        """
        if scale is not None:
            plt.figure(figsize=(6.5*scale, 5*scale))
        else:
            plt.figure(figsize=self.figsize)

        print(self.df[column1].dtype)
        if self.df[column1].dtype != "object":
            raise ValueError("column1 must be category variable")
        if self.df[column2].dtype == "object":
            raise ValueError("column2 must be number variable")

        for uniq in self.df[column2].unique():
            if uniq is np.nan:
                continue

            if uniq in exclude:
                continue


            sns.distplot(self.df[self.df[column1]==uniq][column2], kde=True, rug=False, bins=bins, label=uniq)

        if line is not None:
            if type(line) == list:
                for l in line:
                    plt.axvline(x=l, color="red", linestyle="--")
            else:
                plt.axvline(x=line, color="red", linestyle="--")

        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_category(self, column, scale=None, line=None):
        if scale is not None:
            plt.figure(figsize=(6.5*scale, 5*scale))
        else:
            plt.figure(figsize=self.figsize)

        # 表示メイン
        sns.countplot(self.df[column])

        if line is not None:
            if type(line) == list:
                for l in line:
                    plt.axhline(y=l, color="red", linestyle="--")
            else:
                plt.axhline(y=line, color="red", linestyle="--")
        plt.tight_layout()
        plt.show()
    
    def plot_number(self, column, bins=10, scale=1.0, line=None):
        if scale is not None:
            plt.figure(figsize=(6.5*scale, 5*scale))
        else:
            plt.figure(figsize=self.figsize)

        # 表示メイン
        sns.distplot(self.df[column], kde=True, rug=False, bins=bins)

        if line is not None:
            if type(line) == list:
                for l in line:
                    plt.axvline(x=l, color="red", linestyle="--")
            else:
                plt.axvline(x=line, color="red", linestyle="--")
        plt.tight_layout()
        plt.show()

    def plot_compare_category_category(self, column1, column2, scale=1.0, line=None):
        if scale is not None:
            plt.figure(figsize=(6.5*scale, 5*scale))
        else:
            plt.figure(figsize=self.figsize)
        sns.countplot(x=column1, hue=column2, data=self.df)

        if line is not None:
            if type(line) == list:
                for l in line:
                    plt.axhline(y=l, color="red", linestyle="--")
            else:
                plt.axhline(y=line, color="red", linestyle="--")

        plt.legend()
        plt.title(column2)
        plt.tight_layout()
        plt.show()

    def plot_float_float(
            self, 
            column1, 
            column2, 
            hue=None,
            scale=1.0,
        ):
        if scale is not None:
            scale = self.scale

        # 表示メイン
        sns.jointplot(column1, column2, data=self.df, hue=hue, palette='Set2', size=6*scale)

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