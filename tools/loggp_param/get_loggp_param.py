import os
import sys
import glob

import pandas as pd
from sklearn.linear_model import LinearRegression


def process_file(file_path) -> pd.DataFrame:
    """
    Process a single CSV file to extract loggp parameters.
    This function should be implemented to handle the specific logic
    for processing the CSV files.
    """
    print(f"Processing file: {file_path}")
    
    # Read the CSV file
    raw_df = pd.read_csv(file_path, sep='\s+')

    # Create new DataFrame for parameters
    param_df = pd.DataFrame(columns=["s", "L", "o_s", "o_r", "g", "G"]) 

    ## Fit raw data to obtain g and G parameters ##
    # Create new column size-1
    raw_df["size-1"] = raw_df["size"] - 1
    # Create new column (PRTT(n,0,s) - PRTT(n,0,1)) / (n-1)
    raw_df["PRTT_diff"] = (raw_df["PRTT(n,0,s)"] - raw_df["PRTT(1,0,s)"]) / (raw_df["n"] - 1)
    # Fit linear regression to obtain g and G parameters
    X = raw_df[["size-1"]].values.reshape(-1, 1)
    y = raw_df["PRTT_diff"].values 
    model = LinearRegression()
    model.fit(X, y)
    # Extract parameters
    G = model.coef_[0]
    g = model.intercept_
    rsquared = model.score(X, y)

    print(f"Extracted parameters: g = {round(g,15)}, G = {round(G, 15)} ({round(1/G/(1024**3), 5)} GB/s), r^2 = {round(rsquared,5)}")

    # Other parameters
    param_df["s"] = raw_df["size"]
    param_df["L"] = raw_df["PRTT(1,0,s)"].iloc[0] / 2
    param_df["o_s"] = (raw_df["PRTT(n,d,s)"].values - raw_df["PRTT(1,0,s)"].values) / (raw_df["n"].values - 1) - raw_df["d"].values
    param_df["o_r"] = (raw_df["PRTT_server(n-1,d,s)"].values - raw_df["d"].values * (raw_df["n"].values - 1)) / (raw_df["n"].values - 1)
    param_df["g"] = g
    param_df["G"] = G
    

    return param_df

def main():
    # Get the home directory from the environment variable PICCL_HOME
    home_dir = os.getenv("PICCL_HOME")
    if not home_dir:
        print("Error: PICCL_HOME environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    
    # Construct the path to data/osu_loggp_raw
    data_dir = os.path.join(home_dir, "data", "osu_loggp_raw")
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Get the list of CSV files in the directory
    files = glob.glob(os.path.join(data_dir, "*.csv"))

    # Create output directory if it doesn't exist
    output_dir = os.path.join(home_dir, "data", "loggp_param")
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    for file_path in files:
        df = process_file(file_path)
        df.to_csv(os.path.join(output_dir, os.path.basename(file_path)), index=False, float_format='%.15e', sep=",")

    



if __name__ == "__main__":
    main()