import pathlib
from DataMethods import DataLoader 
from sklearn.model_selection import train_test_split

def split_data(data):
    df = data.copy()
    train,test = train_test_split(df,test_size=0.2,random_state=42)
    return train,test

def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent

    data_path = home_dir.as_posix() + "/data/raw/"
    output_path = home_dir.as_posix() + "/data/raw/"
    data = DataLoader.load_data(data_path,"online_shoppers_intention.csv")

    train,test = split_data(data)
    DataLoader.save_data(output_path,"online_shoppers_intention_train.csv",train)
    DataLoader.save_data(output_path,"online_shoppers_intention_test.csv",test)


if __name__ == "__main__":
    main() 
