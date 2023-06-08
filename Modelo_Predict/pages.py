from st_pages import Page, show_pages, add_page_title
import streamlit as st


if 'selected_page' not in st.session_state:
    st.session_state['selected_page']=None
    
page1=Page("Preprocesado1.py","🧹 Limpieza de datos")
page2=Page("Preprocesado2.py","🔄 Transformacion de variables")
page3=Page("train_test.py","🔍 Variables entrenamientos y pruebas")
page4=Page("modelos.py","📈 Elaboracion modelos")


pages=[page1,page2,page3,page4]
show_pages(pages)

selected_pages=st.session_state['selected_page']

if selected_pages is not None:
    exec(open(selected_pages.filename).read())