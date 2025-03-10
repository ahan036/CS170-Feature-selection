import numpy as np 
import urllib.request 

url = 'https://github.com/ahan036/CS170-Feature-selection/blob/main/CS170_Small_Data__49.txt'
urllib.request.urlretrieve(url, 'CS170_Small_Data_49.txt')
data = np.loadtxt('CS170_Small_Data_49.txt')


#pseudocode from the slides 
'''def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    for i in range(data.shape[0]):
        object_to_classify = data[i, 1:]
        label_object_to_classify = data[i, 0]
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        
        for k in range(data.shape[0]):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k, 1:]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
            
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy '''

#data = our file 
#current_set = the set of features we are selecting 
#feature_to_add = the feature we might add, we have to test the accuracy first 
#test our seach
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    accuracy = np.random.rand() 
    return accuracy

#more code from the slides : project_2 briefing
def feature_search(data):
    current_set_of_features = []
    for i in range(data.shape[1] - 1):
        print(f'On the {i + 1}th level of the search tree')
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        
        for k in range(data.shape[1] - 1):
            if not set(current_set_of_features).intersection({k}):
                print(f'consider adding the {k + 1} feature')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1)
                
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f'On level {i + 1} i added feature {feature_to_add_at_this_level + 1}')