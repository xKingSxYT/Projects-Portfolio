import streamlit as st
import pandas as pd
import numpy as np

st.title('Limpieza de datos')
def import_data():
    tipo_archivo = st.selectbox('Selecciona el tipo de archivo que deseas importar', ['CSV', 'XLSX'])

    if 'CSV' in tipo_archivo:
        csv = st.file_uploader("Importe su archivo CSV")
        df = pd.read_csv(csv)
    elif 'XLSX' in tipo_archivo:
        xlsx = st.file_uploader("Importe su archivo XLSX")
        df = pd.read_excel(xlsx)
    else:
        st.warning("Por favor, selecciona un archivo CSV.")
        
    return df

def preprocess_data(df):
    st.subheader('Realizar preprocesado de datos')
    tipos_datos = st.multiselect('Selecciona los tipos de datos a mostrar', list(df.dtypes.unique()))
    if tipos_datos:
        st.write(df.select_dtypes(include=tipos_datos).head(2))
        st.write('Valores nulos y descriptivos')
        tipos_datos_null = st.selectbox('Selecciona los tipos de datos', list(df.dtypes.unique()))
        st.dataframe(df.select_dtypes(include=tipos_datos_null).isnull().sum())
        st.dataframe(df.select_dtypes(include=tipos_datos_null).describe())

    if 'filled_columns' not in st.session_state:
        st.session_state.filled_columns = []

    fill_method = st.selectbox('Rellenar los valores nulos con: ', ('Mean', 'Mode', 'Booleanos'))
    null=df.isnull().mean()
    cat_bool= null[null>0.25].index
    df_bool=df[cat_bool]
    df_modified=df.drop(cat_bool,axis=1)
    fill_dtype=st.multiselect('Selecciona los tipos de datos que quieres corregir valores nulos',
                                list(df.dtypes.unique()))

    if st.sidebar.button('Aplicar cambios'):
        df_modified = df_modified.copy()
        df_bool_filled=df_bool.copy()

        for col in df_modified.columns:
            if df_modified[col].dtype in fill_dtype:
                if fill_method == 'Mean':
                    df_modified[col].fillna(df_modified[col].mean(), inplace=True)
                elif fill_method == 'Mode':
                    df_modified[col].fillna(df_modified[col].mode()[0], inplace=True)
                else:
                  for col in df_bool_filled.columns:    
                    df_bool_filled[col].fillna('None',inplace=True)
                

        df_combined=pd.concat([df_modified,df_bool_filled],axis=1)
        df.update(df_combined)
        df_combined.to_csv('train_processed.csv', index=False)
        st.dataframe(df_combined)


        st.session_state.df_modified = df_modified

        if st.sidebar.button('Guardar cambios'):
            df_modified.to_csv('train_processed.csv', index=False)
            st.sidebar.write('Cambios guardados correctamente.')

    if st.sidebar.button('Abrir archivo guardado'):
        if 'df_modified' in st.session_state:
            df_saved = st.session_state.df_modified
            st.dataframe(df_saved)
        else:
            st.sidebar.write('No hay cambios guardados.')

def main():

    st.write('Importa tu archivo')

    if 'df' not in st.session_state:
        st.session_state.df = import_data()

    st.dataframe(st.session_state.df.head(1))

    if st.sidebar.button('Reiniciar'):
        st.session_state.df = import_data()

    preprocess_data(st.session_state.df)

if __name__ == '__main__':
    main()
