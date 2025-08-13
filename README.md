# YUTU-ROBOT

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="sHIQ93CY23E9Pn5JqnMg")
project = rf.workspace("yutu-uhfcs").project("yutu-3pkx9")
version = project.version(1)
dataset = version.download("yolov11")
                


https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolo11-object-detection-on-custom-dataset.ipynb
                
