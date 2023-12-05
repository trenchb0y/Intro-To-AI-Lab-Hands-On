import pandas as pd
import numpy as np

# Load Iris Dataset
dwn_url='https://drive.google.com/uc?id=' + '1C5-s_COuWrjn52wzs-QtEYc6QE_IlR2O'
df = pd.read_csv(dwn_url)

df = df.drop('Id', axis=1) # Menghapus kolom 'Id' dari dataframe

# Shuffle Data
df = df.sample(150).reset_index(drop=True)

# Melakukan folding pada data
fold1 = (df.iloc[0:50].reset_index(drop=True), df.iloc[50:150].reset_index(drop=True)) 
fold2 = (df.iloc[50:100].reset_index(drop=True), pd.concat([df.iloc[0:50], df.iloc[100:150]]).reset_index(drop=True))
fold3 = (df.iloc[100:150].reset_index(drop=True), df.iloc[0:100].reset_index(drop=True))

test, train = fold2
print(train) # Menampilkan data fold2 untuk training sebanyak 100 baris

# Normalizations
# Melakukan normalisasi data dengan min-max normalization
def norm(df):
  df =  (df - df.min()) / (df.max() - df.min())
  return df

X = df.drop('Species', axis=1) # Menghapus kolom species(karena akan kita gunakan sebagai class)
y = df.Species 

X = norm(X) # Melakukan normalisasi untuk data fitur
print(X)

X.describe()

# Menghitung jarak menggunakan euclidean distance
def euclidean(x1, x2):
  return # formula for find euclidean distance

# Euclidean distance dari data baris pertama dan kedua
euclidean(X.iloc[0], X.iloc[1])

# Training KNN
def knn(X_train, y_train, X_test, k): # k sebagai banyaknya neighbors yang ditentukan
  dist = []
  # Menghitung distance dari data training dan data testing
  for row in range(X_train.shape[0]):
    dist.append(euclidean(X_train.iloc[row], X_test))
  
  data = X_train.copy() 
  data['Dist'] = dist        # Menambahkan data distance pada data
  data['Class'] = y_train    # Menambahkan class pada data
  data = data.sort_values(by= 'Dist').reset_index(drop=True) # Mengurutkan data berdasarkan distance

  y_pred = data.iloc[:k].Class.mode() # Mengambil label kelas yang paling sering muncul diantara k-NN 
  return y_pred[0]

# Menghitung akurasi dari output berdasarkan label kelas
def acc(y_pred, y_true):
  true = 0
  for i in range(len(y_pred)):
    if y_pred[i] == y_true[i]:
      true+=1
  return true/len(y_pred)

# Evaluasi model dengan menggunakan data fold
def evaluate(fold, k):
  test, train = fold # what data we should use here
  X_train, y_train = train.drop('Species', axis=1), train.Species # what column we want to drop
  X_test, y_test = test.drop('Species', axis=1), test.Species # what column we want to drop
  X_train = norm(X_train) # normalize
  X_test =  norm(X_test)# normalize
  y_preds = []
  for row in range(X_test.shape[0]):
    y_preds.append(knn(X_train, y_train, X_test.iloc[row], k)) # the data, the data label, X_test.iloc[row], assign to choose how many point that are close
  
  return (acc(y_preds, y_test))

k = 5
accs = []
folds = [fold1, fold2, fold3]
for i in range(len(folds)):
  accs.append(evaluate(folds[i], k))
print(f'Menggunakan k : {k}, dengan rata-rata akurasi : {sum(accs)/3}')