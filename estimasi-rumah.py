import pickle
import streamlit as st

model = pickle.load(open('estimasi-rumah.sav', 'rb'))

st.title('Estimasi Harga Rumah di India')
# ['', '', '', '', '', '', '', '', '']
area = st.number_input('Masukkan luas area',min_value=1650.0, max_value=16200.0, step=0.1)

bedrooms = st.number_input('Masukkan jumlah kamar tidur',min_value=1.0, max_value=6.0, step=1.0)

bathrooms = st.number_input('Masukkan jumlah kamar mandi',min_value=1.0, max_value=4.0, step=1.0)

stories = st.number_input('Masukkan jumlah tingkat lantai rumah',min_value=1.0, max_value=4.0, step=1.0)

parking = st.number_input('Masukkan jumlah Jumlah tempat parkir yang tersedia',min_value=0.0, max_value=3.0, step=1.0)

mainroad = st.selectbox(
    'Apakah rumah terletak di dekat jalan utama?', ['Yes', 'No'])
if mainroad == 'Yes':
    mainroad = 1
else:
    mainroad = 0

guestroom = st.selectbox(
    'Apakah rumah memiliki kamar tamu?', ['Yes', 'No'])
if guestroom == 'Yes':
    guestroom = 1
else:
    guestroom = 0

basement = st.selectbox(
    'Apakah rumah memiliki ruang bawah tanah?', ['Yes', 'No'])
if basement == 'Yes':
    basement = 1
else:
    basement = 0

hotwaterheating = st.selectbox(
    'Apakah rumah memiliki pemanas air?', ['Yes', 'No'])
if hotwaterheating == 'Yes':
    hotwaterheating = 1
else:
    hotwaterheating = 0

airconditioning = st.selectbox(
    'Apakah rumah memiliki AC?', ['Yes', 'No'])
if airconditioning == 'Yes':
    airconditioning = 1
else:
    airconditioning = 0

prefarea = st.selectbox(
    'Apakah rumah terletak di area pilihan?', ['Yes', 'No'])
if prefarea == 'Yes':
    prefarea = 1
else:
    prefarea = 0

furnishingstatus = st.selectbox(
    'Bagaimana Status furnishing rumah?', ['furninished', 'semi-furnished', 'unfurnished'])
if furnishingstatus == 'furninished':
    furnishingstatus = 2
elif furnishingstatus == 'semi-furnished':
    furnishingstatus = 1
else:
    furnishingstatus = 0

predict = ''

if st.button('Estimasi Harga'):
    predict = model.predict(
        [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]]
    )
    st.write('estimasi harga rumah dalam INR : ', predict)
    st.write('estimasi harga rumah dalam INR : ', predict * 190,75)