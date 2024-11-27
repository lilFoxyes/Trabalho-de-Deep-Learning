import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class Analise_do_trabalho_de_quarta_feira_que_nome_preguicoso_do_cao:
    def __init__(self):
        pass

    def get_data(self):
        url = "https://www.ssp.df.gov.br/violencia-contra-a-mulher/"
        response = requests.get(url)
        html = soup(response.content, 'html.parser')
        link = html.find('a', href=lambda href: href and "wp-conteudo/uploads" in href and href.endswith(".xls"))
        link = link['href']
        df = pd.read_excel(link)
        linha = df.stack().loc[lambda x: x == "CRIMES DE VIOLÊNCIA DOMÉSTICA NO DF - ÚLTIMOS SEIS ANOS"].index[0][0]
        df.columns = df.iloc[linha]
        df = df.iloc[linha + 1:]
        for i, col in enumerate(df.columns):
            if pd.isna(col):
                df.columns.values[i] = df.iloc[0, i]
        df = df[1:]
        df = df.loc[:, ~df.columns.isna()]
        df = df.melt(id_vars=["CIDADE"], var_name="Ano", value_name="Quantidade")
        df["Ano"] = pd.to_datetime(df["Ano"], format='%Y', errors='coerce')
        df.dropna(subset=["Ano"], inplace=True)
        df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors='coerce')
        df.dropna(subset=["Quantidade"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.df = df
        return self.df

    def plot_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Please run get_data() first.")
        
        unique_cities = self.df["CIDADE"].unique()
        
        for city in unique_cities:
            city_data = self.df[self.df["CIDADE"] == city]
            plt.figure(figsize=(14, 8))
            sns.barplot(data=city_data, x="Ano", y="Quantidade", ci=None)
            plt.title(f"Index of Woman Violence in {city} by Year")
            plt.xlabel("Year")
            plt.ylabel("Quantity")
            plt.xticks(rotation=45)
            
            # Add numbers above each bar
            for p in plt.gca().patches:
                plt.gca().annotate(format(p.get_height(), '.1f'), 
                                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha = 'center', va = 'center', 
                                   xytext = (0, 9), 
                                   textcoords = 'offset points')
            
            plt.tight_layout()
            plt.show()
 

    def forecast(self):
        if self.df is None:
            raise ValueError("Data not loaded. Please run get_data() first.")
        
        # Prepare the data
        self.df['Year'] = self.df['Ano'].dt.year
        X = self.df[['Year', 'CIDADE']]
        X = pd.get_dummies(X, columns=['CIDADE'])
        y = self.df['Quantidade']

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

        # Make predictions for the next year
        next_year = self.df['Year'].max() + 1
        future_data = pd.DataFrame({'Year': [next_year] * len(self.df['CIDADE'].unique()), 'CIDADE': self.df['CIDADE'].unique()})
        future_data = pd.get_dummies(future_data, columns=['CIDADE'])
        future_data = future_data.reindex(columns=column_names, fill_value=0)

        # Convert future_data to numpy array and ensure the data type is float
        future_data = future_data.values.astype(np.float32)

        predictions = model.predict(future_data)
        future_data_df = pd.DataFrame(future_data, columns=column_names)
        future_data_df['Quantidade'] = predictions

        print(future_data_df)



def main():
    bot = Analise_do_trabalho_de_quarta_feira_que_nome_preguicoso_do_cao()
    bot.get_data()
    #bot.plot_data()
    bot.forecast()


if __name__ == "__main__":
    main()