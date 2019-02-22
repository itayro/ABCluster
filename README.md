# ABCluster

1. need to determine the form of measurement for the experiments:
   in the article they used CEP (classification error percentage) which is 100 x (misclassified examples) / (size of test set)
   i think it will be nice to add:
   - time to run the algorithm
   - check imbalanced data
   - check data with high dimentiality (number of features)
   - check large/small data

2. need to update the parameter options:
   - num_of_dimensions : instead of all or one, we should check how the algorithm works with varing number of dimensions when
                         generateing new sample (employee and onlooker phase)
   - employee_to_onlooker_ratio: we need to check what is the needed ratio between the number of employee bees to onlooker bees
   - ...
3. need to search for other objective functions relevent for clustering problem that may produce better results
4. need a generic way of showing the clusters (2D & 3D - optional)
5. need to ensure that the training set contains all the labels
6. more stuff:
   - choosing smaller number of features (choosing 5 features out of 13 for instance)
   - ... 
