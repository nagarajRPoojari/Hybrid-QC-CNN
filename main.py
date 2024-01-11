import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
from HybridQCCNN.config.configuration import ConfigurationManager
from HybridQCCNN.pipeline import Pipeline , merge
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Hybrid Classical Quantum CNN")

@st.cache_resource()
def load_model():
    config=ConfigurationManager().get_model_trainer_config()
    pipeline=Pipeline(config=config)
    return pipeline


pipeline=load_model()


c1,c2=st.columns(2)

selected_country = c2.selectbox("Select model", ["Resnet50", "Hybrid-quantum-Resnet50"])


if selected_country == "Resnet50":
    device = ["GPU"]
elif selected_country == "Hybrid-quantum-Resnet50":
    device = ["qiskit.aer", "default.qubit"]
else:
    device = []
    

selected_city = c2.selectbox("Select device", device)


image=c1.file_uploader("Upload image",type=["jpg",'png','jpeg'])
if image !=None and selected_country!=None:
    res,prob=pipeline.inference(image)
    c1.write(res)
    c1.image(image, caption="Uploaded Image",width=200)

    c2.bar_chart(prob,x='class')

ops=["classical Resnet50 vs Hybrid Resnet50 ","classical Resnet18 vs Hybrid Resnet18 ","classical inception vs Hybrid inception "]
graph=st.selectbox("select training graph",options=ops)

col1,col2=st.columns(2)


resnet50 = "results/resnet50.csv"
resnet50 = pd.read_csv(resnet50)


hybrid_resnet50 = "results/Hybridresnet50.csv"
hybrid_resnet50 = pd.read_csv(hybrid_resnet50)


resnet18 = "results/resnet18.csv"
resnet18 = pd.read_csv(resnet18)


hybrid_resnet18 = "results/HybridResnet18.csv"
hybrid_resnet18 = pd.read_csv(hybrid_resnet18)


inceptionv3 = "results/inceptionV3.csv"
inceptionv3 = pd.read_csv(inceptionv3)


hybrid_inceptionv3 = "results/HybridInceptionV3.csv"
hybrid_inceptionv3 = pd.read_csv(hybrid_inceptionv3)



    
    

col1.write("Training")
col2.write("validation")
if graph==ops[0]:
    a,b=merge(resnet50,hybrid_resnet50,'resnet50')
    
    col1.line_chart(a)
    col2.line_chart(b)
elif graph==ops[1]:
    a,b=merge(resnet18,hybrid_resnet18,'resnet18')
    
    col1.line_chart(a)
    col2.line_chart(b)

elif graph==ops[2]:
    a,b=merge(inceptionv3,hybrid_inceptionv3,'InceptionV3')
    
    col1.line_chart(a)
    col2.line_chart(b)
    
