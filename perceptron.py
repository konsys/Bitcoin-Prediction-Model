from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

df = pd.read_csv('bitcoin.csv')
df.head()

splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date']) 

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)

X_train = features[:200]
y_train = target[:200]

X_test = features[201:]
y_test = target[201:]

# print(X_test)


# 3. Initialize the Perceptron model
# random_state is set for reproducibility
model = Perceptron(eta0=0.03, max_iter=1000, tol=1e-2, shuffle=False)

# 4. Train the Perceptron model using the training data
model.fit(X_train, y_train)

# # 5. Make predictions on the test set
y_pred = model.predict(X_test[:])

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


dlrsAmount = 1000
btAmount = 0.5
closePrice = 0
list = []
for index, row in df[220:].iterrows():
    pr = model.predict([row[['open-close', 'low-high', 'is_quarter_end']]])

    predicted = pr[:][0]
    closePrice = row['Close']
    if(predicted == 1):
        if dlrsAmount:
            list.append(dlrsAmount)
       
            btAmount = btAmount + dlrsAmount / closePrice
            dlrsAmount = 0
    else:
        if btAmount:
            dlrsAmount = btAmount * closePrice
            btAmount = 0
        # if btAmount > 0 or dlrsAmount > 0:
        #     print('dlrsAmount', dlrsAmount)
        #     print('btAmount', btAmount)
        #     print()
    
df = pd.DataFrame(list)

print(df.describe())

print(1, dlrsAmount)
print(2, btAmount)
print(3, btAmount * closePrice)