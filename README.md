<h1><p align="center">Head Detector using YoloV8</p></h1>

<p align="center">
  <a href="https://github.com/AbelKidane-abita/Reports/blob/main/notebooks/Report.ipynb"><img  alt="Static Badge" src="https://img.shields.io/badge/Report-Jupyter%20Notebook-orange" target="_blank">
   <a  href="https://huggingface.co/spaces/AbelKidane/headdetector" ><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow" target="_blank"> 
</p> 
     
<p align="center">
  This a head detection model based on Ultralytics YoloV8 model. Please clone the repository to use it or use the Hugging Face deployment.
</p>

<h2>Documentation</h2>

### 1. Install requirements

Install dependencies using the following command
```
pip install -r requirements.txt

```

### 2. Run inference from CLI and show the results using Matplotlib
```
python ./predict_image.py --image_path=image/test.jpg

```

### 3. Run inference using Streamlit
```
streamlit run ./Streamlit_test.py

```


