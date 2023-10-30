import pickle
import streamlit as st

model = pickle.load(open('mobil_bekas.sav', 'rb'))

st.title('Estimasi Harga Jual Mobil Bekas')

Year = st.selectbox("Input Tahun Mobil", [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018])
Present_Price = st.number_input("Input Harga Jual Saat Ini (Lakh INR)", value=9.85)
Owner = st.selectbox("Input Jumlah Pemilik Sebelumnya", [0,1,2,3])
Kms_Driven = st.number_input("Input Jarak Tempuh", min_value=0)
Age = st.number_input("Input Usia Kendaraan", min_value=0)

predict = ''

if st.button('Estimasi Harga Jual Mobil Bekas'):
      predict = model.predict(
            [[Year, Present_Price, Owner, Kms_Driven, Age]]
      )
st.write ('Estimasi Harga Jual Mobil Bekas Dalam EUR : ', predict)
st.write ('Estimasi Harga Jual Mobil Bekas Dalam IDR (Juta) :  ', predict*19000)
