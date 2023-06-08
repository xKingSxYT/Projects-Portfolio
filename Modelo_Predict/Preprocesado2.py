import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

st.title('Transformacion de variables')


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

    drop = st.multiselect('Selecciona las variables que deseas eliminar de tu modelo', df.columns)
    df_modified = df.drop(drop, axis=1)
    st.dataframe(df_modified.head(2))

    object_columns = df_modified.select_dtypes(include=['object'])
    columns = st.multiselect('Selecciona las columnas para generar las variables dummy', object_columns.columns)

    if columns:
        encoding_method = st.radio('Selecciona el método para transformar', ['LabelEncoder', 'Dummies'])

        if encoding_method == 'LabelEncoder':
            df_modified_copy = df_modified.copy()
            label_encoder = LabelEncoder()
            for col in columns:
                df_modified_copy[col] = label_encoder.fit_transform(df_modified_copy[col])
        elif encoding_method == 'Dummies':
            df_modified_copy = df_modified.copy()
            dummies_df = pd.get_dummies(df_modified_copy[columns], prefix_sep=';', drop_first=True)
            df_modified_copy = pd.concat([df_modified_copy, dummies_df], axis=1)

        st.dataframe(df_modified_copy.head(5))

        if st.sidebar.button('Aplicar cambios'):
            st.session_state.df_modified = df_modified_copy.copy()
            st.session_state.changes_applied = True

            if 'second_transformation' not in st.session_state:
                st.session_state.second_transformation = False

            if st.sidebar.button('Transformar otra variable'):
                preprocess_data(st.session_state.df_modified)

            elif not st.session_state.second_transformation:
                st.session_state.second_transformation = True
                remaining_columns = list(set(object_columns.columns) - set(columns))
                remaining_columns.sort()
                if len(remaining_columns) > 0:
                    st.write('Transformación con el método restante')
                    remaining_method = 'Dummies' if encoding_method == 'LabelEncoder' else 'LabelEncoder'
                    remaining_encoding_columns = st.multiselect('Selecciona las columnas para la transformación',
                                                                remaining_columns)
                    if len(remaining_encoding_columns) > 0:
                        preprocess_remaining_data(st.session_state.df_modified, remaining_encoding_columns,
                                                  remaining_method)

    if st.sidebar.button('Guardar cambios') and st.session_state.changes_applied:
        st.session_state.df_modified.to_csv('train_processed.csv', index=False)
        st.sidebar.write('Cambios guardados correctamente.')


def preprocess_remaining_data(df, columns, encoding_method):
    if encoding_method == 'LabelEncoder':
        df_modified_copy = df.copy()
        label_encoder = LabelEncoder()
        for col in columns:
            df_modified_copy[col] = label_encoder.fit_transform(df_modified_copy[col])
    elif encoding_method == 'Dummies':
        df_modified_copy = df.copy()
        dummies_df = pd.get_dummies(df_modified_copy[columns], prefix_sep=';', drop_first=True)
        df_modified_copy = pd.concat([df_modified_copy, dummies_df], axis=1)

    st.dataframe(df_modified_copy.head(5))

    if st.sidebar.button('Aplicar cambios'):
        st.session_state.df_modified = df_modified_copy.copy()
        st.session_state.changes_applied = True



def main():

    st.write('Importa tu archivo')

    if 'df' not in st.session_state:
        st.session_state.df = import_data()


    if st.sidebar.button('Reiniciar'):
        st.session_state.df = import_data()

    preprocess_data(st.session_state.df)


if __name__ == '__main__':
    if 'changes_applied' not in st.session_state:
        st.session_state.changes_applied = False
    main()