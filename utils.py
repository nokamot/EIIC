import numpy as np
import pandas as pd
import pickle

# Extract lower triangle of correlation matrix and get rid of diagonal elements
def flatten_cors(cors):
    num_element = cors.shape[0]
    flatten = np.zeros(int((num_element*(num_element-1))/2))
    i_ = 0
    for i in range(num_element):
        for j in range(i):
            flatten[i_] = cors[i,j]
            i_ += 1
            
    return flatten

# Save trained models by pickle
def save_models(models, output_dir, site):
    with open(output_dir+'/result_models_'+site+'.bf', 'wb') as resultsf:
        pickle.dump(models, resultsf)

# Calculate performance indices by confusion matrix
def model_measurements(matrix):
    acc = (matrix[0,0] + matrix[1,1])/np.sum(matrix)

    recall = (matrix[0,0])/(matrix[0,0]+matrix[0,1])

    specificity = (matrix[1,1])/(matrix[1,0]+matrix[1,1])

    ppv = (matrix[0,0])/(matrix[0,0]+matrix[1,0])
    
    npv = (matrix[1,1])/(matrix[0,1]+matrix[1,1])

    return acc, recall, specificity, ppv, npv

# Output the array of performance indices as xlsx file
def save_result_csv(results, index, column, output_dir):
    with pd.ExcelWriter(output_dir+'/results.xlsx') as writer:
        for model in results:
            result = results[model]
            df = pd.DataFrame(result, index=index, columns=column)
            df.to_excel(writer, sheet_name=model)
            
