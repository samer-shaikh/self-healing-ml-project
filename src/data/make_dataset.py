import pathlib
from DataMethods import DataLoader 

def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent

    data_path = home_dir.as_posix() + "/data/raw/"
    data = DataLoader.load_data(data_path,"online_shoppers_intention.csv")
    print(data.shape)

if __name__ == "__main__":
    main() 
