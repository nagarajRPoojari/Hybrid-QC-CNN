import streamlit as st 
import pandas as pd
from HybridQCCNN.config.configuration import ConfigurationManager
from HybridQCCNN.pipeline import Pipeline


def load_model():
    configer=ConfigurationManager()
    config=configer.get_model_trainer_config()
    pipeline = Pipeline(config=config)
    return pipeline


pipeline=load_model()


st.title("""
    Hi , How are you        
""")

df=pd.read_csv('./results/resnet18.csv')


uploaded_file = st.file_uploader("Choose an Image", type=["jpg","png"])

if uploaded_file is not None:

    ans=pipeline.inference(img_path=uploaded_file)
    st.write(ans)

else:
    st.info("Please upload an Image.")
st.line_chart(df)