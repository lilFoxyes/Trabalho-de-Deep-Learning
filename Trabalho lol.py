import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as soup
import kagglehub
import os
import tempfile
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



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


    def forecast(self):
        if self.df is None:
            raise ValueError("Data not loaded. Please run load_data() first.")
        
        # Prepare the data
        self.df['Year'] = self.df['release_date'].dt.year
        X = self.df[['Year', 'class']]
        X = pd.get_dummies(X, columns=['class'])
        y = self.df['champion'].groupby(self.df['Year']).transform('count')

        # Save the column names before converting to numpy array
        column_names = X.columns

        # Convert to numpy arrays and ensure the data type is float
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Build the model
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}')

        # Make predictions for 2023 and 2024
        future_years = [2023, 2024]
        future_data = pd.DataFrame({'Year': future_years})
        future_data = future_data.reindex(columns=column_names, fill_value=0)

        # Convert future_data to numpy array and ensure the data type is float
        future_data = future_data.values.astype(np.float32)

        predictions = model.predict(future_data)
        future_data_df = pd.DataFrame(future_data, columns=column_names)
        future_data_df['Predicted_Champions'] = predictions

        print(future_data_df)


    
def main():
    lol = League_Of_Legends()
    lol.get_data()
    lol.transform_data()
    lol.plot_data()
    lol.forecast()

if __name__ == '__main__':
    main()