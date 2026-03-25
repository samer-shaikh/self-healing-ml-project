import pandas as pd
import os

class DataLoader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_data(path:str,name:str) -> pd.DataFrame:

        try:
            data = pd.read_csv(path+name)
            return data
        
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
        
        

    @staticmethod
    def save_data(path:str,name:str,data:pd.DataFrame) -> str:

        try:
            data.to_csv(os.path.join(path,name),index=False)
            return "data save successful"
        
        except Exception as e:
            raise Exception(f"Error saving data: {e}")
        

        