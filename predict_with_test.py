import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.preprocessing import MultiLabelBinarizer
from lightgbm import LGBMRegressor


data = pd.read_csv('test_int20.csv')
model = pd.read_pickle('lgmb_model.pkl')

def fill_na(df):
    
    df['book_genre'] = df['book_genre'].fillna('None')
    df['book_format'] = df['book_format'].fillna('None')
    
    return df

def make_norm_columns(df):
    
    df['author_list'] = df['book_authors'].apply(lambda x: x.split('|'))
    df['author_count'] = df['author_list'].apply(lambda x: len(x))
    df['genre_list'] = df['book_genre'].apply(lambda x: x.split('|') if x!='None'
                                             else [])
    df['genre_count'] = df['genre_list'].apply(lambda x: len(x))
    
    df['book_pages'] = df['book_pages'].fillna('0 pages')
    df['book_pages'] = df['book_pages'].apply(lambda x: re.findall(r"\d+", x)[0]).astype(int)
    
    return df

def make_autors_info(x, at_dict):
    

    workcount = []
    fan_count = []
    rating_count = []
    review_count = []
    average_rate = [] 
    
    for author in x:
        if author in at_dict.keys():
            workcount.append(at_dict[author].get('workcount'))
            fan_count.append(at_dict[author].get('fan_count'))
            rating_count.append(at_dict[author].get('rating_count'))
            review_count.append(at_dict[author].get('review_count'))
            average_rate.append(at_dict[author].get('average_rate'))

    return {'workcount': np.mean(workcount),
           'fan_count': np.mean(fan_count),
           'rating_count': np.mean(rating_count),
           'review_count': np.mean(review_count),
           'average_rate': np.mean(average_rate)}

def add_author_info(df):
    
    author_data = pd.read_csv('final_dataset.csv')
    author_data = author_data.drop_duplicates('name')
    author_data = author_data.set_index('name')
    at_dict = author_data.to_dict(orient='index')
    
    author_inf = df['author_list'].apply(lambda x: make_autors_info(x, at_dict))
    author_inf = pd.DataFrame(author_inf.tolist())
    df = pd.concat([df, author_inf], axis=1)
    df['desc_len'] = df['book_desc'].apply(lambda x: len(x.split()))
    
    return df
    
def preprocess_books(column):

    column = column.fillna('None').apply(lambda x: set(re.split(r"\|", x)))

    mlb = MultiLabelBinarizer()
    expandedLabelData = mlb.fit_transform(column)
    labelClasses = mlb.classes_

    expandedLabels = pd.DataFrame(expandedLabelData, columns=labelClasses)
    
    return expandedLabels


def preproc(data_pred):
    
    data_pred = fill_na(data_pred)
    data_pred = make_norm_columns(data_pred)

    data_pred = add_author_info(data_pred)
    train_cols = ['book_pages','book_review_count','book_rating_count', 
              'author_count', 'genre_count', 'workcount', 'fan_count', 'rating_count', 'review_count', 'average_rate']

    test_crop = data_pred[train_cols]
    genres_test = preprocess_books(data['book_genre'])
    genres_cols = pd.read_csv('select_genres.csv')['select_genres'].tolist()
    genres_test = genres_test[genres_cols]
    test_crop = pd.concat([test_crop, genres_test.iloc[data.index, :]], axis=1)
    
    return  test_crop

def select_predict_sample():
    st.subheader("Оберіть бажаний екземпляр")
    st.markdown("За допомогою слайдеру нижче оберіть заданий екземпляр тестового датасету, аби отримати передбачення моделі.")
    predict_id = st.slider('Номер екземпляру', min_value=0, max_value=3019)
    st.write("Ви обрали наступний екземпляр: ", predict_id)
    st.write(data.iloc[predict_id, :])
    return predict_id

def make_predict(pred_id):
    
    value = st.button('Отримати прогноз')
    if value:
        with st.spinner('Очікуйте результати...'):
            
            data_temp = data.copy()
            preproc_values = preproc(data_temp)
            preproc_value = np.array(preproc_values.iloc[pred_id, :]).reshape(1, -1)
            
            #st.write("Проведений препроцессинг: ")
            #st.write(preproc_value)
            pred = model.predict(preproc_value)
            st.subheader("**Прогноз сформовано!** Отримані результати наступні:")
            st.write("Очікувана оцінка книги складає ", np.round(pred[0], 2), "балів.")
            
            return pred[0]



