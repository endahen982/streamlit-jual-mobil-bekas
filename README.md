# Laporan Proyek Machine Learning
### Nama : Endah Anisah Fauziyah
### Nim : 211351050
### Kelas : TIF Pagi B

![mobil_bekas](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/1c0e1319-0628-4701-8ce8-9a9d7ea47c34)


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
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/6aebe364-fc9f-463d-a1de-59cf3de08f3a)

Hasil grafik ini adalah tampilan visual yang menggambarkan sejauh mana kolom-kolom dalam data frame "df" bisa berkorelasi satu sama lain. Kolom yang memiliki korelasi positif tinggi akan terlihat sebagai sel dengan warna yang lebih terang, sedangkan kolom yang berkorelasi negatif tinggi akan terlihat sebagai sel dengan warna yang lebih gelap.


```bash
Fuel_Types = df.groupby('Fuel_Type').count()[['Selling_Price']].sort_values(by='Selling_Price',ascending=True).reset_index()
Fuel_Types = Fuel_Types.rename(columns={'Selling_Price':'car data'})
```

```bash
fig = plt.figure(figsize=(20,5))
sns.barplot(x=Fuel_Types['Fuel_Type'], y=Fuel_Types['car data'], color='maroon')
plt.xticks(rotation=40)
```
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/151357a2-8401-4f7b-abb7-c228e28e6a77)

Hasil grafik Jumlah Mobil Bekas Berdasarkan Jenis Bahan Bakar menunjukkan perbandingan jumlah data mobil yang menggunakan jenis bahan bakar yang berbeda. Berdasarkan grafik ini, kita dapat mengamati hal berikut:

- Bahan bakar 'CNG' memiliki jumlah data mobil paling sedikit, ditunjukkan oleh batang pertama pada grafik.
- Bahan bakar 'Diesel' memiliki jumlah data mobil yang sedang, ditunjukkan oleh batang kedua pada grafik.
- Bahan bakar 'Petrol' memiliki jumlah data mobil paling banyak, ditunjukkan oleh batang terakhir pada grafik.

Hasil ini memberikan tentang distribusi data mobil berdasarkan jenis bahan bakar berupa gambaran visual tentang seberapa banyak mobil yang tersedia untuk masing-masing jenis bahan bakar. Disini bisa dilihat bahwa Jenis bahan bakar dengan batang grafik yang lebih tinggi memiliki lebih banyak mobil yang tersedia, sedangkan jenis bahan bakar dengan batang yang lebih pendek memiliki jumlah mobil yang lebih sedikit. Ini akan membantu preferensi konsumen terkait dengan jenis bahan bakar kendaraan.

```bash
Years = df.groupby('Year').count()[['Selling_Price']].sort_values(by='Selling_Price',ascending=True).reset_index()
Years = Years.rename(columns={'Selling_Price':'Count'})
```

```bash
fig = plt.figure(figsize=(20,5))
sns.barplot(x=Years['Year'], y=Years['Count'], color='maroon')
```
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/46675d0e-d7ca-45f4-8007-d1c4e2efd121)

Hasil grafik Jumlah Data Penjualan Berdasarkan Tahun bisa dilihat bahwa penjualan terbanyak terjadi ditahun 2015 dan penjualan terendah terjadi pada 2004.

```bash
plt.figure(figsize=(15,5))
sns.distplot(df['Present_Price'])
```
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/86f30846-aa01-48f5-8fa5-2e33e96b0edf)

```bash
plt.figure(figsize=(15,5))
sns.distplot(df['Selling_Price'])
```
![image](https://github.com/endahen982/streamlit-jual-mobil-bekas/assets/148830351/28bcfd8a-e831-4512-9f04-c2824bb0efd3)

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
Dibagian ini saya menambahkan data tahun mobil, yang bermaksud sebagai sudah berapa tahun mobil ini terpakai sejak pembelian pertama.

```bash
df["Selling_Price"].fillna(0, inplace=True)
df["Car_Name"].fillna("No Car_Name", inplace=True)
```
Lalu Mengubah data Float menjadi Integer
```bash
df["Selling_Price"] = df["Selling_Price"].astype("int")
```

```bash
df["Present_Price"] = df["Present_Price"].astype("int")
```

```bash
numerical = []
catgcols = []

for col in df.columns:
    if df[col].dtype== 'int64' :
      numerical.append(col)
    else:
      catgcols.append(col)

for col in df.columns:
    if col in numerical:
      df[col].fillna(df[col].median(), inplace=True)
    else:
      df[col].fillna(df[col].mode()[0], inplace=True)
```

```bash
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in catgcols:
  df[col] = le.fit_transform(df[col])
```

```bash
df.info()
```
Lalu saya mengubah data object menjadi integer yang bertujuan agar dapat memasukan kedalam atribut pada bagian seleksi fitur

```bash
df.head()
```

```bash
features = ['Year','Present_Price','Owner','Kms_Driven','Age','Fuel_Type','Car_Name','Seller_Type']
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
X = df[['Year','Present_Price','Owner','Kms_Driven','Age','Fuel_Type','Car_Name','Seller_Type']]
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
#year=2017, present_price=9.85, owner=0, kms_driven=6900, age=12, Fuel_Type=2 ,Car_Name=68 , Seller_Type=0
input_data = np.array([[2017,9.85,0,6900,12,2,68,0]])

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
