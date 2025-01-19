# CRC_XGB_Pred
XGB based machine learning model for the prediction of activity on colorectal  cancer cell line (KM-12)

##  1. Environment Setup for Running the Models** <br />
Python = 3.7.3 <br />
MolVS = 0.1.1 <br />
standardiser = 0.1.9 <br />
matplotlib = 3.3.4 <br />
numpy       =  1.17.3 <br />
pandas       = 1.2.1 <br />
Babel       =  2.7.0 <br />
scikit-learn = 0.23.2 <br />
scipy        = 1.4.1 <br />
xgboost      = 1.1.1 <br />

##  2. Data preparation
NCI-60 data preparation.

Steps to run the script:
1. Download the Dataprocessing.py file.
2. Create a directory named **Dataset** in you current working directory.
3. Download the NCI-60 and chemical dataset into your **Dataset** directory using link below.

NCI-60 dataset download link: https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Data+Download+-+Previous+Releases?preview=/147193864/374736079/NCI60_GI50_2016b.zip.

Chemical data download link: https://wiki.nci.nih.gov/display/NCIDTPdata/Chemical+Data?preview=/155844992/339380766/Chem2D_Jun2016.zip

4. Unzip both the downloaded files.
5. Activate conda env created using Environment Setup.
4. Once all the files are ready, run the followint script: **python processing.py**
6. The output of this scipt will be automatically store into **Dataset/All_tested_molecules.csv** and cell line data per cell line will be saved in **cell_line_smile**.
7. The **cell_line_smile** directory will have 60 .csv files, each for the one cell line. The colums names are: [NLOGGI50, NSC, SMILE, molecular features]

## 3. Machine learning model building
