import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def initialize_parameters(n_features):
  W = np.zeros(n_features)
  b = 0
  return W, b

def forward_propagation(X, W, b):
  Z = np.dot(X, W) + b
  A = sigmoid(Z)
  return A

def compute_cost(A, Y):
    m = len(Y)
    error = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return error

def backward_propagation(X, A, Y):
    m = len(Y)
    dZ = A - Y
    dW = (1 / m) * np.dot(X.T, dZ)
    db = (1 / m) * np.sum(dZ)
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def logistic_regression(X, Y, num_iterations, learning_rate):
    start_time = time.time()
    n_features = X.shape[1]
    W, b = initialize_parameters(n_features)

    for i in range(num_iterations + 1):
        A = forward_propagation(X, W, b)
        cost = compute_cost(A, Y)
        dW, db = backward_propagation(X, A, Y)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        if i % 100 == 0:
            print(f"Giá trị mất mát sau {i} lần huấn luyện: {cost}")

    end_time = time.time()
    training_time = end_time - start_time

    print("Thời gian huấn luyện: ", training_time,"s")

    print("Cost: ", cost)
    print("Weight: ", W)
    print("Bias: ", b)

    return W, b

def predict(X, W, b):
    A = forward_propagation(X, W, b)
    #predictions = np.round(A)
    predictions = A
    return predictions

input_cols = ['TBKTOnline', 'TBKTTuluan', 'Nghi', 'Online', 'TX1', 'TX2', 'GK']
output_col = 'Actual'

data_train = pd.read_csv('train_data.csv', encoding='utf-8')
X_train = data_train.loc[:, input_cols]
y_train = data_train.loc[:, output_col]

data_test = pd.read_csv('test_data.csv', encoding='utf-8')
X_test = data_test.loc[:, input_cols]
y_test = data_test.loc[:, output_col]

missing_values_train = data_train.isnull().sum()
missing_values_test = data_test.isnull().sum()
print("Số lượng giá trị thiếu trong mỗi cột:")
print(missing_values_train, missing_values_test)

data_filled_train = data_train.fillna(0)
data_filled_test = data_test.fillna(0)

selected_data = data_train[input_cols + [output_col]]
for col in input_cols:
    plt.figure(figsize=(8, 6))
    plt.bar(selected_data[col], selected_data[output_col], width=0.5, align='center')
    plt.xlabel(col)
    plt.ylabel('Giá trị thực tế')
    plt.title(f'Biểu đồ cột cho {col}')
    plt.grid(True)
    plt.show()

epochs = 5000
learning_rate = 0.2
W, b = logistic_regression(X_train, y_train, epochs, learning_rate)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(input_cols, W, color='blue', alpha=0.7, label='Weight')
ax.bar('Bias', b, color='red', alpha=0.7, label='Bias')
ax.set_ylabel('Giá trị')
ax.set_title('Weights và Bias sau khi huấn luyện')
ax.legend()
plt.show()

y_pred = predict(X_test, W, b)

y_test = pd.Series(y_test)
y_pred = pd.Series(y_pred)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(y_test)), y_test, color='blue', label='Giá trị thực tế', alpha=0.7)
ax.scatter(range(len(y_pred)), y_pred, color='red', label='Giá trị dự đoán', alpha=0.7)
for i in range(len(y_test)):
    ax.plot([i, i], [y_test.iloc[i], y_pred.iloc[i]], color='green', alpha=0.7)
ax.set_xlabel('Mẫu thử')
ax.set_ylabel('Giá trị dự đoán')
ax.set_title('Giá trị thực tế và dự đoán của thử nghiệm')
ax.legend()
plt.show()

y_test.reset_index(drop=True, inplace=True)
y_pred.reset_index(drop=True, inplace=True)
accuracy = np.mean(y_pred == y_test)
print(f'Độ chính xác: {accuracy:.2f}')

for pred, actual in zip(y_pred, y_test):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

print("\n\n\n---------------------------------------- TIENG ANH K14  -----------------------------------------")
datak14 = pd.read_csv('TAK14.csv', encoding='utf-8')

X_K14 = datak14.loc[:, input_cols]
Y_K14 = datak14.loc[:, output_col]

y_pred_K14 = predict(X_K14, W, b)

for pred, actual in zip(y_pred_K14, Y_K14):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(range(len(Y_K14)), Y_K14, color='blue', label='Giá trị thực tế', alpha=0.7)
ax.scatter(range(len(y_pred_K14)), y_pred_K14, color='red', label='Giá trị dự đoán', alpha=0.7)
for i in range(len(Y_K14)):
    ax.plot([i, i], [Y_K14[i], y_pred_K14[i]], color='green', alpha=0.7)
ax.set_xlabel('Mẫu thử')
ax.set_ylabel('Giá trị dự đoán')
ax.set_title('Giá trị thực tế và dự đoán của Khóa 14')
ax.legend()
plt.show()

print("\n\n\n---------------------------------------- TIENG ANH K15  -----------------------------------------")
datak15 = pd.read_csv('TAK15.csv', encoding='utf-8') #mã hóa utf-8

X_K15 = datak15.loc[:, input_cols]
Y_K15 = datak15.loc[:, output_col]

y_pred_K15 = predict(X_K15, W, b)

for pred, actual in zip(y_pred_K15, Y_K15):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(range(len(Y_K15)), Y_K15, color='blue', label='Giá trị thực tế', alpha=0.7)
ax.scatter(range(len(y_pred_K15)), y_pred_K15, color='red', label='Giá trị dự đoán', alpha=0.7)
for i in range(len(Y_K15)):
    ax.plot([i, i], [Y_K15[i], y_pred_K15[i]], color='green', alpha=0.7)
ax.set_xlabel('Mẫu thử')
ax.set_ylabel('Giá trị dự đoán')
ax.set_title('Giá trị thực tế và dự đoán của Khóa 15')
ax.legend()
plt.show()

print("\n\n\n---------------------------------------- TIENG ANH K16  -----------------------------------------")
datak16 = pd.read_csv('TAK16.csv', encoding='utf-8')

X_K16 = datak16.loc[:, input_cols]
Y_K16 = datak16.loc[:, output_col]

y_pred_K16 = predict(X_K16, W, b)

for pred, actual in zip(y_pred_K16, Y_K16):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(range(len(Y_K16)), Y_K16, color='blue', label='Giá trị thực tế', alpha=0.7)
ax.scatter(range(len(y_pred_K16)), y_pred_K16, color='red', label='Giá trị dự đoán', alpha=0.7)
for i in range(len(Y_K16)):
    ax.plot([i, i], [Y_K16[i], y_pred_K16[i]], color='green', alpha=0.7)
ax.set_xlabel('Mẫu thử')
ax.set_ylabel('Giá trị dự đoán')
ax.set_title('Giá trị thực tế và dự đoán của Khóa 16')
ax.legend()
plt.show()

print("\n\n\n---------------------------------------- TIENG ANH K17  -----------------------------------------")
datak17 = pd.read_csv('TAK17.csv', encoding='utf-8') #mã hóa utf-8

X_K17 = datak17.loc[:, input_cols]
Y_K17 = datak17.loc[:, output_col]

y_pred_K17 = predict(X_K17, W, b)

for pred, actual in zip(y_pred_K17, Y_K17):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(range(len(Y_K17)), Y_K17, color='blue', label='Giá trị thực tế', alpha=0.7)
ax.scatter(range(len(y_pred_K17)), y_pred_K17, color='red', label='Giá trị dự đoán', alpha=0.7)
for i in range(len(Y_K17)):
    ax.plot([i, i], [Y_K17[i], y_pred_K17[i]], color='green', alpha=0.7)
ax.set_xlabel('Mẫu thử')
ax.set_ylabel('Giá trị dự đoán')
ax.set_title('Giá trị thực tế và dự đoán của Khóa 17')
ax.legend()
plt.show()