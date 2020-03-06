# Network Classification Models

* ICA 
* Weighted Random Forest


The ICA implementation is largely based on and adapted from: [https://github.com/tkipf/ica](https://github.com/tkipf/ica). Changed were made to use networkx and pandas. 

## Installation

```bash
python setup.py install
```

## Requirements
* sklearn
* networkx

## Run the code
To run the code, by using a python notebook run either of the functions. 
```bash
train.run_ICA(model_name, model_parameter, n_iterations)
train.run_WQRF(n_estimators)
```
Check the `Untitled.ipynb` file for an example.

## Data
### ICA
In order to use your own data, you have to provide these two files in `utils.py`
* an N by N adjacency matrix (N is the number of nodes, labels should be included and defined in the code) to replace:
```bash 
data/adj_matrix_28Nov.csv
```
* an N by D feature matrix (D is the number of features per node) to replace:
```bash 
data/real-data-28Nov.csv
```

### WQRF
In order to use your own data, you have to provide these two files in `train.py`
* an N by D feature matrix (D is the number of features per node) to replace:
```bash 
data/real-data-28Nov.csv
```
* an D by 1 feature importance matrix (D is the number of features per node) to replace:
```bash 
data/feat_importance_28_Nov.csv
```

# This example
The current data is data retrieved from [https://github.com/Guillemdb/Happiness-inside-a-job-PyDataBcn17](https://github.com/Guillemdb/Happiness-inside-a-job-PyDataBcn17)

It is employee churn data and the adjacency matrix was rendered from the all_features_all.csv.