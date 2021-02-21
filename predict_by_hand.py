import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.preprocessing import MultiLabelBinarizer


data = pd.read_csv('test_int20.csv')
model = pd.read_pickle('lgmb_model.pkl')
train_info = pd.read_csv('train_info.csv')
test_info = pd.read_csv('test_info.csv')

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
    print(author_inf)
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
              'author_count', 'genre_count', 'workcount', 'fan_count', 'rating_count', 'review_count', 'average_rate', 'height', 'width']

    test_crop = data_pred[train_cols]
    
    genres_test = preprocess_books(data['book_genre'])
        
    genres_cols = pd.read_csv('select_genres.csv')['select_genres'].tolist()
    genres_test = genres_test[genres_cols]
    test_crop = pd.concat([test_crop, genres_test.iloc[data.index, :]], axis=1)
    
    return  test_crop

def make_data():
    st.subheader("Створимо екземпляр власноруч")
    st.markdown("За допомогою полей нижче оберіть всі параметри бажаного екземпляру.")
    bk_ft = ['Kindle Edition', 'Paperback', 'Mass Market Paperback',
       'Hardcover', 'ebook', 'Nook', 'Board book', 'Bonded Leather',
       'Trade Paperback', 'Library Binding', 'Audio CD', 'Audible Audio',
       'online fiction', 'Unknown Binding', 'Interactive ebook',
       'Capa mole - 15,5 x 23 x 2cm', 'Comic', 'Audiobook',
       'Online Fiction - Complete', 'Audio Cassette', 'Broché',
       'Leather Bound', 'eBook Kindle', 'paper', 'hardcover', 'Poche',
       'audio cassette', 'Board Book', 'PDF ', 'Diary', 'Bolsillo',
       'Taschenbuch', 'Online Fiction', 'paperback', 'Other Format',
       'Mass Market', 'Perfect Paperback', 'Audio', 'Box-Set',
       'free online', 'Softcover', 'Broschiert', 'Hardcover Chapbook',
       'Klappenbroschur', 'Paperback, Kindle, Ebook, Audio', 'Innbundet',
       'Wattpad', 'Bìa mềm', 'Capa Mole', 'Hardcover Boxed Set',
       'Digital Comic', 'Box Set', 'Hardback', 'Boxed Set', 'MP3 CD',
       'softcover', 'Paperback, eBook', 'cloth', 'Trade Paper',
       'Hard Cover', 'Podiobook', 'web', 'Large Paperback',
       'Tapa dura con sobrecubierta', 'Novel', 'Capa mole',
       'Graphic Novels', 'Spiral-bound', 'Turtleback', 'Board',
       'Newsprint', 'Tapa blanda', 'CD-ROM', 'Flexibound',
       'paperback, Kindle eBook', 'Cofanetto', 'broché',
       'School & Library Binding', 'Bantam New Fiction',
       'Hardcover, Case bound', 'Mass Market Paperback ',
       'Paperback and Kindle', 'Paperback ', 'Vinyl Cover', 'Audio Cd',
       'Slipcased Hardcover', 'Kovakantinen', 'Novelty Book', 'Hardbound',
       'Kindle', 'Science Fiction Book Club Omnibus', ' Trade Paperback',
       'Pocket', 'Audio Book', 'capa mole', 'Capa Dura', 'Edición Kindle',
       'Big Book', 'Capa comum', 'Brossura', 'Gebunden', '単行本',
       'Paperback/Ebook', 'Podcast', 'Fiction', 'Softcover, free ebook',
       'Pamphlet', 'Hardcover-spiral', 'audiobook', 'Casebound',
       'Hard cover, Soft cover, e-book', 'Capa dura', 'Unbound',
       'Newsletter Serial', 'Trade paperback', 'online serial',
       'Hardcover im Schuber', 'Gebundene Ausgabe',
       'Interactive Fiction, Gamebook', 'hardbound', 'Digital',
       'Brochura', 'revised edition']
    #['book_format', 'book_pages','book_review_count','book_rating_count', 
              #'author_count', 'genre_count', 'workcount', 'fan_count', 'rating_count', 'review_count', 'average_rate']
    book_title = st.text_input("Поле book_title", 'The Ship')
    url_500 = pd.read_csv('url_shape.csv')['book_image_url'].unique().tolist()[:25]
    book_image_url = st.selectbox("Поле book_image_url", url_500)   
    book_desc = st.text_area('Поле book_desc', 'About')
    all_genres = pd.read_csv('all_genres.csv')
    all_genres = all_genres['0'].unique().tolist()
    genres = st.multiselect('Поле book_genre', all_genres, ['Fiction', 'Fantasy', 'Romance'])
    genres = '|'.join(genres)
    all_authors = pd.read_csv('all_authors.csv')['0'].unique().tolist()[:25]
    authors = st.multiselect('Поле book_authors', all_authors, ['A. Buelow'])
    authors = '|'.join(authors)
    book_format = st.selectbox("Поле book_format", bk_ft)
    book_pages = st.slider('Поле book_pages', min_value=1, max_value=1500)
    book_pages = str(book_pages) + ' pages'
    book_review_count = st.slider('Поле book_review_count', min_value=1, max_value=150_000)
    book_rating_count = st.slider('Поле book_rating_count', min_value=1, max_value=5_000_000)
    
    
    df2 = {'id': [np.max(data.id)], 'book_title': [book_title], 'book_image_url': [book_image_url],
           'book_desc': [book_desc], 'book_genre': [genres], 'book_authors': [authors], 'book_format': [book_format],
           'book_pages': [book_pages], 'book_review_count': [book_review_count], 'book_rating_count': [book_rating_count]}

    return pd.DataFrame(df2)

def make_predict_hand(pred_df):
    
    value = st.button('Отримати прогноз')
    if value:
        with st.spinner('Очікуйте результати...'):
            
            #sample_value = data.iloc[pred_id, :]
            data_temp = pred_df.copy()
            data_temp['url_shape'] = train_info['url_shape'][4]
            data_temp['url_shape'] = data_temp['url_shape'].apply(lambda x: eval(x))
            
            data_temp['height'] = data_temp['url_shape'].apply(lambda x: x[0] if x else None)
            data_temp['width'] = data_temp['url_shape'].apply(lambda x: x[1] if x else None)
            data_temp.drop(columns=['url_shape'], inplace=True)
            #data_temp.iloc[np.max(data.id), :] = pred_df.iloc[:, :]
            preproc_values = preproc(data_temp)
            

            preproc_value = np.array(preproc_values.iloc[0, :]).reshape(1, -1)
            pred = model.predict(preproc_value)

            st.subheader("**Прогноз сформовано!** Отримані результати наступні:")
            st.write("Очікувана оцінка книги складає ", np.round(pred[0], 2), "балів.")

            
            return pred[0]



