import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


def accuracy(a, b, i):
    count = 0
    for j in range(0, len(a)):
        if -i < (a[j] - b[j]) < i : count += 1
        else: continue
    return (count/len(a))*100


tabledf = pd.read_csv("data.csv")
tableX = tabledf.drop(columns="wage_per_hour")
tableY = tabledf[["wage_per_hour"]]

model = Sequential()
n_cols = tableX.shape[1]
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping_monitor = EarlyStopping(patience=5)
model.fit(tableX, tableY, validation_split=0.2, epochs=75, callbacks=[early_stopping_monitor])
test_y_predictions = model.predict(tableX)

i = 0
while True:
    acc = accuracy(tableY.values.tolist(), test_y_predictions, i)
    print(f"i = {i}\nacc = {acc}%")
    if acc >= 80:
        break
    i += 0.2
