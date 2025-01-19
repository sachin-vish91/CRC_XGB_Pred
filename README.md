# CRC_XGB_Pred
XGBoost-based machine learning model for predicting activity on the colorectal cancer cell line (KM-12).

##  1. Environment Setup <br />
```Python          3.7.3``` <br />
```MolVS           0.1.1``` <br />
```standardiser    0.1.9``` <br />
```matplotlib      3.3.4``` <br />
```numpy           1.17.3``` <br />
```pandas          1.2.1``` <br />
```Babel           2.7.0``` <br />
```scikit-learn    0.23.2``` <br />
```scipy           1.4.1``` <br />
```xgboost         1.1.1``` <br />

##  2. NCI-60 data preparation
- Create a Python environment following the instructions in the Environment Setup guide.
  
- Download the ```processing.py``` script.

- In your current working directory, create a folder named ```Dataset```.

- Download the following datasets into the Dataset directory:

   - [NCI-60 dataset](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Data+Download+-+Previous+Releases?preview=/147193864/374736079/NCI60_GI50_2016b.zip) <br />
   - [Chemical Dataset](https://wiki.nci.nih.gov/display/NCIDTPdata/Chemical+Data?preview=/155844992/339380766/Chem2D_Jun2016.zip)

- Unzip both downloaded files into the ```Dataset``` directory.

- Execute the following command to process the datasets:<br />
   ```python processing.py```


## 3. Machine learning model training (XGBoost)

To train the machine learning model, use the following script:<br />
   ```python ../script/XGBboost_MTL_training.py```


## 4. Prediction with trained XGBoost model

After the model training is complete, use the following script to make predictions on a new dataset:<br />
   ```python ../script/prediction.py```

**Note**: Make sure to update the file paths in the script as required for your dataset and environment.
