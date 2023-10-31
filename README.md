# Laporan Proyek Machine Learning
### Nama : Endah Anisah Fauziyah
### Nim : 211351050
### Kelas : TIF Pagi B

## Domain Proyek
Pada hasil dari proyek pembuatan aplikasi "Estimasi Harga Jual Mobil Bekas" menggunakan Streamlit ialah menghadirkan solusi bagi para pengguna yang ingin menjual mobil bekas dan membantu pemilik mobil bekas agar mendapatkan perkiraan harga yang realistis, yang dapat mempermudah pada proses penjualan mobil bekas. Dengan memanfaatkan algoritma machine learning. Linear regression merupakan salah satu model regresi dalam machine learning yang banyak digunakan untuk memodelkan hubungan antara variabel dependen atau variabel independen.

## Business Understanding

### Problem Statements

- Beberapa pemilik mobil bekas mungkin merasa kesulitan dalam menentukan harga jual yang pas untuk mobil mereka. Mereka perlu cara atau sesuatu yang cepat dan akurat untuk memperkirakan harga berdasarkan karakteristik mobil yang akan dijual.

### Goals

- Membuat aplikasi "Estimasi Harga Jual Mobil Bekas" menggunakan streamlit, dimana pemilik mobil dapat memasukkan data seperti tahun, harga saat ini, jumlah pemilik sebelumnya, total jarak tempuh dan usia mobil untuk mendapatkan estimasi harga jual yang akurat.

### Solution statements
- Pengembangan platform Estimasi Harga Jual Mobil Bekas berbasis web adalah solusi untuk memprediksi harga jual mobil bekas tersebut berdasarkan data input karakteristik mobil yang dimasukkan si pemilik mobil.
- Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Pada hasil proyek ini saya menggunakan dataset yang tersedia di kaggle.com
 [Vehicle dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data?select=car+data.csv)  

Untuk bagian ini, kita akan menggunakan teknik EDA


 ```bash
df.columns
```

```bash
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
```

```bash
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/869170d2-1cb6-4eab-92ba-b407a0908161)
```

```bash
Fuel_Types = df.groupby('Fuel_Type').count()[['Selling_Price']].sort_values(by='Selling_Price',ascending=True).reset_index()
Fuel_Types = Fuel_Types.rename(columns={'Selling_Price':'car data'})
```

```bash
fig = plt.figure(figsize=(20,5))
sns.barplot(x=Fuel_Types['Fuel_Type'], y=Fuel_Types['car data'], color='maroon')
plt.xticks(rotation=40)

![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/d6a1b957-70fc-4599-a6ae-e04f8b1d2e33)
```

```bash
Years = df.groupby('Year').count()[['Selling_Price']].sort_values(by='Selling_Price',ascending=True).reset_index()
Years = Years.rename(columns={'Selling_Price':'Count'})
```

```bash
fig = plt.figure(figsize=(20,5))
sns.barplot(x=Years['Year'], y=Years['Count'], color='maroon')
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/bae3134f-5e50-4ae8-b1b7-d60b9718e41a)
```

```bash
plt.figure(figsize=(15,5))
sns.distplot(df['Present_Price'])
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/b8602edf-8f24-42f3-adc4-207821264cc9)

```

```bash
plt.figure(figsize=(15,5))
sns.distplot(df['Selling_Price'])
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/14baa8bc-71b0-4fa7-b536-203e5bcf8c14)
```
### Variabel-variabel pada Vehicle dataset adalah sebagai berikut:
- Car_Name : Mempresentasikan nama merek atau identifikasi mobil.

- Year : Mempresentasikan usia atau umur kendaraan.

- Selling_Price : Merupakan variabel yang mengukur harga jual atau nilai jual. Dengan kata lain, variabel ini dapat menggambarkan berapa harga yang diinginkan oleh pemilik kendaraan saat ingin menjualnya.

- Present_Price : Mempresentasikan harga saat ini pada kendaraan tersebut dan bisa menjadi faktor penting dalam menilai atau membandingkan harga kendaraan bekas.

- Kms_Driven : Kolom atau atribut yang digunakan untuk mengukur jumlah kilometer yang telah ditempuh oleh kendaraan tersebut.

- Fuel_Type : variabel yang mempresentasikan jenis bahan bakar yang digunakan oleh kendaraan tersebut.

- Seller_Type : Jenis penjual atau pemasok yang menjual mobil tersebut.

- Transmission : Mempresentasikan jenis transmisi yang digunakan dalam kendaraan tersebut.

- Owner : Menggambarkan jumlah pemilik sebelumnya dari kendaraan tersebut.

## Data Preparation
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Vehicle dataset, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

## Data Discovery And Profiling

Pertama kita mengimport semua library yang dibutuhkan

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

lalu dilanjut dengan memasukkan file .csv yang telah diextract pada sebuah variable, dan melihat data paling atas dari datasetnya

```bash
df = pd.read_csv('vehicle-dataset-from-cardekho/car data.csv')
```

```bash
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
```

```bash
df.info()
```

```bash
df.head()
```

```bash
sns.heatmap(df.isnull())
```

```bash
print(df.shape)
```

```bash 
import datetime
date_time = datetime.datetime.now()
print(date_time)
df['Age']=date_time.year - df['Year']
```

```bash
df.head()
```

```bash
features = ['Year','Present_Price','Owner','Kms_Driven','Age']
x = df[features]
y = df['Selling_Price']
x.shape, y.shape
```

```bash
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```

## Modeling

```bash
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```

```bash
score = lr.score(X_test, y_test)
print('akurasi model regresi liniear =', score)
```

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini adalah Mean Squared Error (MSE) dan R-squared(R^2). Berdasarkan metrik evaluasi yang digunakan dengan menggunakan Mean Squared Error (MSE) maka semakin rendah MSE, semakin baik model dalam memprediksi harga mobil bekas. Dan apabila menggunakan R-squared(R^2) maka semakin mendekati 1 nilai R^2, semakin baik model dalam menjelaskan variasi dalam harga mobil bekas.

menggunakan MSE dan R^2 adalah untuk mengevaluasi sejauh mana model linear regression proyek kita dapat memprediksi harga jual mobil bekas. Semakin rendah MSE dan semakin tinggi R^2 maka semakin baik performa model proyek kita.

```bash
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

```bash
X = df[['Year','Present_Price','Owner','Kms_Driven','Age']]
y = df['Selling_Price']
```

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```bash
model = LinearRegression()
model.fit(X_train, y_train)
```

```bash
y_pred = model.predict(X_test)
```

```bash
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
```

## Inputan Model Regresi Linear

```bash
#year=2017, present_price=9.85, owner=0, kms_driven=6900, age=12
input_data = np.array([[2017,9.85,0,6900,12]])

prediction = lr.predict(input_data)
print('Estimasi harga mobil dalam EUR :', prediction)
```

```bash
import pickle
filename = 'mobil_bekas.sav'
pickle.dump(lr,open(filename,'wb'))
```

## Deployment

[Estimasi Harga Jual Mobil Bekas](https://app-estimasi-mobil-bekas-5pjjkz3v9p6aqhan2omwwz.streamlit.app/)

![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/83b4e9eb-457b-4b8c-8ff8-89fb84875f06)
