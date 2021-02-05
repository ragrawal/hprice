# hprice
An example of building an ensemble model using "baikal" and using "sklearn-pandas" to manage tranformations. 

# Instructions

```python
conda create -n hprice python=3.7
conda activate hprice
pip install -r requirements.txt
pip install jupyter
jupyter notebook
```
Once you have jupyter running, run train.ipynb. This will train the model and serialize it in the reousrces folder
Next, run the test.ipynb file. It will load the serialized model and generate predictions. 
