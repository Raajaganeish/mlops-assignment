from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv("data/raw/housing.csv", index=False)
print("Data loaded and saved to data/raw/housing.csv")
