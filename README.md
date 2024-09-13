<h1><p align="center">Head Detector using YoloV8</p></h1>

Notice: It is OUTDATED.
<p align="center">
  <a href="https://github.com/AbelKidane-abita/Reports/blob/main/notebooks/Report.ipynb"><img  alt="Static Badge" src="https://img.shields.io/badge/Report-Jupyter%20Notebook-orange" target="_blank">
   <a  href="https://huggingface.co/spaces/AbelKidane/headdetector" ><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow" target="_blank"> 
     <a  href="https://hub.docker.com/r/abelkidane/reports" ><img alt="Static Badge" src="https://img.shields.io/badge/docker-abelkidane%2Freports-blue?logo=docker" target="_blank"> 
    
</p> 
     
<p align="center">
  This is a head detection model based on the Ultralytics YoloV8 model. Please clone the repository to use it or use the Hugging Face deployment.
</p>

<h2>Documentation</h2>

### 1. Install requirements

Install dependencies using the following command. The model is implemented using Python version 3.10. 
```
pip install -r requirements.txt

```

### 2. Run inference from CLI and show the results using matplotlib
```
python ./main.py --image_path=image/test.jpg

```

### 3. Run inference using Streamlit
```
streamlit run ./streamlit_webapp.py

```
### 4. Run inference using Streamlit on Docker

Build the image from the Dockerfile:
```
docker build --tag abel_head_detector .
```


Or pull the image from docker hub:
```
docker pull abelkidane/reports
```

Run the image using docker run:
```
docker run -it abel_head_detector streamlit run streamlit_webapp.py --server.port 8080
```

Or using docker compose (not completed yet)
```
docker build --tag abel_head_detector .
docker compose up

```
