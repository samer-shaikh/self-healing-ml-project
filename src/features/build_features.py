from src.data.DataMethods import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import pathlib
import joblib

def column_transformation(data: pd.DataFrame, median_val) -> pd.DataFrame:

    df = data.copy()

    # Total time = strong engagement signal
    df["EngagementScore"] = (
                                df["ProductRelated_Duration"] +
                                df["Informational_Duration"] +
                                df["Administrative_Duration"]
                            )
    
    # Focus on product vs other pages
    df["InteractionRate"] = df["ProductRelated"] / (df["Administrative"] + 1)

    # Detect poor sessions
    df["BounceExitRatio"] = df["BounceRates"] / (df["ExitRates"] + 1e-5)

    # Returning users buy more
    df["IsReturning"] = (df["VisitorType"] == "Returning_Visitor").astype(int)

    # Captures serious buyers
    if median_val is not None:
        df["HighIntent"] = (df["ProductRelated_Duration"] > median_val).astype(int)
    
    # Deep engagement indicator
    df["AvgTimePerPage"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)

    # Business + behavior together
    df["UserQuality"] = (
                            df["PageValues"] * df["ProductRelated_Duration"]
                        )
    df = df.round(2)

    return df

def column_scaling(train:pd.DataFrame,test:pd.DataFrame):

    # fatch the column names
    num_cols = train.select_dtypes(include=["int64", "float64"]).columns
    object_cols = train.select_dtypes(include=["object","string","category"]).columns

    # making the transformer 
    transformer = ColumnTransformer(transformers=[
        ("StandardScaler",StandardScaler(),num_cols),
        ("OHE",OneHotEncoder(handle_unknown="ignore", sparse_output=False),object_cols)
    ],
    remainder="passthrough")

    #scaling the column
    train_arr = transformer.fit_transform(train)
    test_arr = transformer.transform(test)

    feature_names = transformer.get_feature_names_out()

    # naming the right name to the all columns 
    train = pd.DataFrame(train_arr, columns=feature_names) # type: ignore
    test = pd.DataFrame(test_arr, columns=feature_names) # type: ignore

    return train,test,transformer


def main():
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent

    data_path = home_dir.as_posix() + "/data/raw/"
    output_path = home_dir.as_posix() + "/data/processed/"
    
    # loading the dataset
    train_data = DataLoader.load_data(data_path,"online_shoppers_intention_train.csv")
    test_data = DataLoader.load_data(data_path,"online_shoppers_intention_test.csv")

    
    X_train = train_data.drop(columns="Revenue")
    X_test = test_data.drop(columns="Revenue")

    y_train = train_data["Revenue"]
    y_test = test_data["Revenue"]

    median_val = X_train["ProductRelated_Duration"].median()
    train_data = column_transformation(X_train,median_val)
    test_data = column_transformation(X_test,median_val)
    
    train,test,transformer = column_scaling(train_data,test_data)

    train["Revenue"] = y_train
    test["Revenue"] = y_test


    res = DataLoader.save_data(output_path,"online_shoppers_intention_train.csv",train)
    res = DataLoader.save_data(output_path,"online_shoppers_intention_test.csv",test)
    joblib.dump(median_val, f"{home_dir}/models/median.pkl")
    joblib.dump(transformer, f"{home_dir}/models/preprocessor.joblib")
    print(res)
    print(train.columns)

if __name__ == "__main__":
    main()
