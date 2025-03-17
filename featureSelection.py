import numpy as np 
import math #euclidian distance 

#for running the dataset while testing 
def dataset(): 
    dataset = int(input('What dataset do u want large = 1, small = 2'))
    print(dataset)
    if dataset == 1: 
        dataset = 'CS170_Large_Data__87.txt'
    elif dataset == 2: 
        dataset = 'CS170_Small_Data__49.txt'
    else:
        print ('Defaulting to small dataset')
        dataset = 'CS170_Small_Data__49.txt' 
    data = np.loadtxt(dataset)
    return np.shape(data)[0], np.shape(data)[1], data

#to choose the dataset and method 
def main():
    print("Welcome to Ashley's feature selection algorithm.")
    # file_input = input("Type in the name of the file to test: ")
    # print(file_input)
    # data = np.genfromtxt(file_input)
    select_algo = input("Which algorithm should we run? 1) Forward Selection /n 2)Backward Elimination /n")
    instances, features, data = dataset()
    print('This dataset has ' + str(features - 1) + ' features (not including the class attribute), with ' + str(instances) + ' instances.\n')
    print('Running nearest neighbor with all {} features, using \"leaving-one-out\" evalutation, I get an accuracy of {}%\n'.format(data.shape[1]-1, leave_one_out_cross_validation(data)))
    # if select_algo == '1':
    #     forward_selection(data) 
    # else select_algo == '2':
    #     backward_elimination(data)



#def forward_selection(data)
#def backward_elimination(data)


#pseudocode from the slides 
'''def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    #shape tells us the number of rows in our data, basically we are looping for x # of rows 
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
def leave_one_out_cross_validation(data, current_set =None, feature_to_add=None):
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


#run the main menu 
if __name__ == "__main__":
    main()