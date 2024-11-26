import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as soup
import kagglehub
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


class League_Of_Legends:
    def __init__(self):
        self.dataset = kagglehub.dataset_download("jennifermacnaughton/league-of-legends-short-story-analysis")
    
    def get_data(self):
        path = self.dataset
        files = os.listdir(path)
        csv_file_path = os.path.join(path, files[0])
        df = pd.read_csv(csv_file_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file_path = tmp_file.name
            df = pd.read_csv(csv_file_path)
            df.to_csv(tmp_file_path, index=False)
        
        self.df = pd.read_csv(tmp_file_path)
        return self.df

    def transform_data(self):
        df = self.df.copy()
        df.drop(columns=["class_2nd", "sub-class_2nd"], inplace=True)
        df_before_transformation = df.copy()
        df.sort_values(by="release_date", inplace=True)
        df = df[["champion", "release_date", "class"]]
        df["release_date"] = pd.to_datetime(df["release_date"])
        self.df = df.copy()
        return self.df
    

    def plot_data(self):
        df = self.df.copy()
        plt.figure(figsize=(10, 6))
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_date'].dt.year.value_counts().sort_index().plot(kind='bar')
        release_counts = df['release_date'].dt.year.value_counts().sort_index()
        ax = release_counts.plot(kind='bar')
        plt.title('Number of Champions Released per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Champions')
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.show()
        plt.figure(figsize=(10, 6))
        df['class'].value_counts().plot(kind='bar')
        class_counts = df['class'].value_counts()
        ax = class_counts.plot(kind='bar')
        plt.title('Distribution of Champions by Class')
        plt.xlabel('Class')
        plt.ylabel('Number of Champions')
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.show()
        ...


    def forecast_releases(self):
        df = self.df.copy()

        # Aggregate the number of champions released per year
        yearly_counts = df.groupby(df["release_date"].dt.year).size().reset_index(name="champion_count")
        yearly_counts.rename(columns={"release_date": "ds", "champion_count": "y"}, inplace=True)

        print("\nYearly Aggregated Data:")
        print(yearly_counts.head())

        # Ensure ds is datetime and y is float
        yearly_counts["ds"] = pd.to_datetime(yearly_counts["ds"], format='%Y')
        yearly_counts["y"] = yearly_counts["y"].astype(float)

        # Split into training and test sets
        train_size = int(len(yearly_counts) * 0.7)
        train = yearly_counts.iloc[:train_size]
        test = yearly_counts.iloc[train_size:]

        print("\nTraining Data:")
        print(train)
        print("\nTest Data:")
        print(test)

        # Fit the Prophet model on the training set
        model = Prophet()
        model.fit(train)

        # Forecast for the test set
        future = model.make_future_dataframe(periods=len(test), freq='YE')
        forecast = model.predict(future)

        print("\nForecast Data:")
        print(forecast.head())

        # Evaluate the model
        test_predictions = forecast[-len(test):]["yhat"].values
        mae = mean_absolute_error(test["y"].values, test_predictions)
        print(f"\nMean Absolute Error on Test Set: {mae:.2f}")

        # Forecast until 2024
        future_extended = model.make_future_dataframe(periods=(2024 - yearly_counts["ds"].dt.year.max()), freq='YE')
        extended_forecast = model.predict(future_extended)

        # Extract 2023 and 2024 predictions
        future_releases = extended_forecast[extended_forecast["ds"].dt.year >= 2023][["ds", "yhat"]]
        future_releases["yhat"] = future_releases["yhat"].round().astype(int)  # Round to integers
        print(f"\nForecasted Releases:")
        print(future_releases)

        return future_releases
    
    def predict_classes(self, future_releases):
        df = self.df.copy()
        
        # Calculate historical class proportions
        class_counts = df["class"].value_counts(normalize=True)
        print("\nHistorical Class Proportions:")
        print(class_counts)
        
        # Predict class distribution for future releases
        predictions = []
        for _, row in future_releases.iterrows():
            year = row["ds"].year
            total_releases = row["yhat"]
            class_distribution = (class_counts * total_releases).round().astype(int)
            predictions.append({"year": year, "class_distribution": class_distribution.to_dict()})
        
        print("\nPredicted Class Distribution:")
        for pred in predictions:
            print(f"Year {pred['year']}: {pred['class_distribution']}")
        
        return predictions
    
def main():
    lol = League_Of_Legends()
    lol.get_data()
    lol.transform_data()
    lol.plot_data()
    future_releases = lol.forecast_releases()
    lol.predict_classes(future_releases)
if __name__ == '__main__':
    main()