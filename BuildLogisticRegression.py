import pandas as pd #thư viện đọc ghi file
import numpy as np #thư viện ma trận, tính toán, .....
import matplotlib.pyplot as plt #thư viện vẽ đồ thị
import time #thư viện thời gian
from sklearn.model_selection import train_test_split #thư viện chia tệp

def sigmoid(z):
  #Hàm kích hoạt giá trị từ 0 dến 1
  return 1 / (1 + np.exp(-z))

def initialize_parameters(n_features):
  #Khởi tạo các tham số như weight, bias đều bằng 0
  W = np.zeros(n_features)
  b = 0
  return W, b

def forward_propagation(X, W, b):
  #lan truyền thằng và tìm đầu ra
  Z = np.dot(X, W) + b
  A = sigmoid(Z)
  return A

def compute_cost(A, Y):
  #tính giá trị lỗi cho hàm bằng cross-entropy loss
    m = len(Y)
    error = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return error

def backward_propagation(X, A, Y):
  #lan truyền ngược tính đạo hàm riêng
    m = len(Y)
    dZ = A - Y
    dW = (1 / m) * np.dot(X.T, dZ)
    db = (1 / m) * np.sum(dZ)
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
  #update lại weight và bias dùng gradient descent
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def logistic_regression(X, Y, num_iterations, learning_rate):
    start_time = time.time() #bắt đầu tính thời gian huấn luyện
    n_features = X.shape[1] #số lượng đầu vào tx1, tx2, gk, tgonl
    W, b = initialize_parameters(n_features) #khởi tạo các tham số weight, bias bằng 0

    for i in range(num_iterations + 1): #epochs từ i đến n - 1
        A = forward_propagation(X, W, b) #tính đầu vào net và đầu ra cho lan truyền thẳng
        cost = compute_cost(A, Y) # tính lỗi của mô hình
        dW, db = backward_propagation(X, A, Y) #đạo hàm riêng của các tham số
        W, b = update_parameters(W, b, dW, db, learning_rate) #cập nhật lại tham số

        if i % 100 == 0: #in ra lỗi cứ sau 100 lần
            print(f"Giá trị mất mát sau {i} lần huấn luyện: {cost}")

    end_time = time.time()  #kết thúc quá trình huấn luyện
    training_time = end_time - start_time

    # In ra thời gian huấn luyện mô hình
    print("Thời gian huấn luyện: ", training_time,"s")

    # In ra sai số, weight và bias cuối cùng
    print("Cost: ", cost)
    print("Weight: ", W)
    print("Bias: ", b)

    return W, b #trả về ma trận weight và bias

def predict(X, W, b):
    A = forward_propagation(X, W, b) #tính đầu vào net và đầu ra
    #predictions = np.round(A) #làm tròn chạy từ 0 đến 1 sẽ làm tròn <0.5 thì mình quy la 0 còn >= 0.5 mình quy nó 1
    predictions = A # giá trị dự đoán
    return predictions

# Đọc dữ liệu từ tệp CSV bằng pandas
data = pd.read_csv('data_TA1.csv', encoding='utf-8') #mã hóa utf-8

# Chọn các cột đầu vào và cột đầu ra
input_cols = ['TBKTOnline', 'TBKTTuluan', 'Nghi', 'Online', 'TX1', 'TX2', 'GK']
output_col = 'Actual'
X = data.loc[:, input_cols]
y = data.loc[:, output_col]

# Hiển thị số lượng giá trị thiếu trong mỗi cột
missing_values = data.isnull().sum()
print("Số lượng giá trị thiếu trong mỗi cột:")
print(missing_values)

# Cho những giá trị bị thiếu bằng giá trị 0
data_filled = data.fillna(0)

# Vẽ đồ thị cột thể hiện sự tương quan của đầu vào với đầu ra
# selected_data = data[input_cols + [output_col]]
# for col in input_cols:
#     plt.figure(figsize=(8, 6))
#     plt.bar(selected_data[col], selected_data[output_col], width=0.5, align='center')
#     plt.xlabel(col)
#     plt.ylabel('Giá trị thực tế')
#     plt.title(f'Biểu đồ cột cho {col}')
#     plt.grid(True)
#     plt.show()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lưu file train ra ngoài
train_data = pd.DataFrame({'TBKTOnline': X_train['TBKTOnline'], 'TBKTTuluan': X_train['TBKTTuluan'], 'Nghi': X_train['Nghi'], 'Online': X_train['Online'], 'TX1': X_train['TX1'], 'TX2': X_train['TX2'], 'GK': X_train['GK'], 'Actual': y_train})
train_data.to_csv('train_data.csv', index=False, encoding='utf-8')

# Lưu file test ra ngoài
test_data = pd.DataFrame({'TBKTOnline': X_test['TBKTOnline'], 'TBKTTuluan': X_test['TBKTTuluan'], 'Nghi': X_test['Nghi'], 'Online': X_test['Online'], 'TX1': X_test['TX1'], 'TX2': X_test['TX2'], 'GK': X_test['GK'], 'Actual': y_test})
test_data.to_csv('test_data.csv', index=False, encoding='utf-8')

# # Đọc dữ liệu train từ tệp CSV bằng pandas
# data_train = pd.read_csv('train_data.csv', encoding='utf-8') #mã hóa utf-8
# X_train = data_train.loc[:, input_cols]
# y_train_pre = data_train.loc[:, output_col]
#
# # Đọc dữ liệu test từ tệp CSV bằng pandas
# data_test = pd.read_csv('test_data.csv', encoding='utf-8') #mã hóa utf-8
# X_test = data_test.loc[:, input_cols]
# y_test_pre = data_test.loc[:, output_col]

# Huấn luyện mô hình Logistic Regression trên tập huấn luyện
epochs = 5000
learning_rate = 0.2
W, b = logistic_regression(X_train, y_train, epochs, learning_rate)

# Plot trọng số và bias
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(input_cols, W, color='blue', alpha=0.7, label='Weight')
ax.bar('Bias', b, color='red', alpha=0.7, label='Bias')
ax.set_ylabel('Giá trị')
ax.set_title('Weights và Bias sau khi huấn luyện')
ax.legend()
plt.show()

# Dự đoán trên tập kiểm tra
y_pred = predict(X_test, W, b)

#Tạo biểu đồ scatter plot giữa giá trị dự đoán và giá trị thực tế
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

# Đánh giá mô hình trên tập kiểm tra theo độ chính xác phần trăm
y_test.reset_index(drop=True, inplace=True)
y_pred.reset_index(drop=True, inplace=True)
accuracy = np.mean(y_pred == y_test)
print(f'Độ chính xác: {accuracy:.2f}')

# In giá trị dự đoán và giá trị thực để so sánh
for pred, actual in zip(y_pred, y_test):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

print("\n\n\n---------------------------------------- TIENG ANH K14  -----------------------------------------")
# Đọc dữ liệu từ tệp CSV bằng pandas
datak14 = pd.read_csv('TAK14.csv', encoding='utf-8') #mã hóa utf-8

X_K14 = datak14.loc[:, input_cols]
Y_K14 = datak14.loc[:, output_col]

y_pred_K14 = predict(X_K14, W, b)

for pred, actual in zip(y_pred_K14, Y_K14):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

#Tạo biểu đồ scatter plot giữa giá trị dự đoán và giá trị thực tế
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
# Đọc dữ liệu từ tệp CSV bằng pandas
datak15 = pd.read_csv('TAK15.csv', encoding='utf-8') #mã hóa utf-8

X_K15 = datak15.loc[:, input_cols]
Y_K15 = datak15.loc[:, output_col]

y_pred_K15 = predict(X_K15, W, b)

for pred, actual in zip(y_pred_K15, Y_K15):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

#Tạo biểu đồ scatter plot giữa giá trị dự đoán và giá trị thực tế
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
# Đọc dữ liệu từ tệp CSV bằng pandas
datak16 = pd.read_csv('TAK16.csv', encoding='utf-8') #mã hóa utf-8

X_K16 = datak16.loc[:, input_cols]
Y_K16 = datak16.loc[:, output_col]

y_pred_K16 = predict(X_K16, W, b)

for pred, actual in zip(y_pred_K16, Y_K16):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

#Tạo biểu đồ scatter plot giữa giá trị dự đoán và giá trị thực tế
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
# Đọc dữ liệu từ tệp CSV bằng pandas
datak17 = pd.read_csv('TAK17.csv', encoding='utf-8') #mã hóa utf-8

X_K17 = datak17.loc[:, input_cols]
Y_K17 = datak17.loc[:, output_col]

y_pred_K17 = predict(X_K17, W, b)

for pred, actual in zip(y_pred_K17, Y_K17):
    print("Giá trị dự đoán:", pred, "Giá trị thực tế:", actual)

#Tạo biểu đồ scatter plot giữa giá trị dự đoán và giá trị thực tế
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