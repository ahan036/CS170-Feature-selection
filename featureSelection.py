import numpy as np 

#for running the dataset while testing 
#https://www.geeksforgeeks.org/find-the-number-of-rows-and-columns-of-a-given-matrix-using-numpy/
def dataset(): 
    dataset = int(input('What dataset do u want? large = 1, small = 2'))
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
    print("Welcome to Ashley's Feature Selection Algorithm. \n")
    # file_input = input("Type in the name of the file to test: ")
    # print(file_input)
    # data = np.genfromtxt(file_input)
    select_algo = int(input("Type the number of the algorithm you want to run. \n 1) Forward Selection \n 2) Backward Elimination \n"))
    instances, features, data = dataset()

    features = list(range(1, features))
    print('This dataset has ' + str(features) + ' features (not including the class attribute), with ' + str(instances) + ' instances.\n')
    #test to make sure we are considering all features in our test 
    print(features)
    all_features = leave_one_out_cross_validation(data, features, -1)

    print(all_features)
    print('Running nearest neighbor with all' + str(features) +  ' features, using \"leaving-one-out\" evalutation, I get an accuracy of ' + str(all_features * 100) + '%')
    
    if select_algo == 1:
        subset, accuracy = forward_selection(data)
    else:
         subset, accuracy = backward_elimination(data)
    print('Finished search!! The best feature subset is ' + str(subset) +  ' which has an accuracy of ' + str(accuracy * 100) + '%')

#pseudocode from the slides 
def leave_one_out_cross_validation(data, current_set, feature_to_add):

    #this is to make sure we add the new feature to our existing set 
    if (feature_to_add != -1):
        #we need to copy or else we permanently alter current_set and we cant run the original 
        current_set = current_set.copy() 
        current_set.append(feature_to_add)
    
    number_correctly_classified = 0
    #shape tells us the number of rows in our data, basically we are looping for x # of rows 
    for i in range(data.shape[0]): 
        features = len(current_set)
        #Added to code to allow for proper scaling to features
        object_to_classify = data[i, current_set]
        label_object_to_classify = data[i][0]
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf

        for k in range(data.shape[0]):
            if k != i:
                #Added to code to allow for proper scaling to features
                distance = np.sqrt(np.sum((object_to_classify - data[k, current_set]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy

#data = our file 
#current_set = the set of features we are selecting 
#feature_to_add = the feature we might add, we have to test the accuracy first 
#test our seach
# def leave_one_out_cross_validation(data, current_set =None, feature_to_add=None):
#     accuracy = np.random.rand() 
#     return accuracy

#more code from the slides : project_2 briefing
#this will turn into our forward and backward search 
# def feature_search(data):
#     current_set_of_features = []
#     for i in range(data.shape[1] - 1):
#         print(f'On the {i + 1}th level of the search tree')
#         feature_to_add_at_this_level = None
#         best_so_far_accuracy = 0
#         for k in range(data.shape[1] - 1):
#             if not set(current_set_of_features).intersection({k}):
#                 print(f'consider adding the {k + 1} feature')
#                 accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1)
                
#                 if accuracy > best_so_far_accuracy:
#                     best_so_far_accuracy = accuracy
#                     feature_to_add_at_this_level = k
        
#         current_set_of_features.append(feature_to_add_at_this_level)
#         print(f'On level {i + 1} i added feature {feature_to_add_at_this_level + 1}')

def forward_selection(data):
    current_set_of_features = []
    solution_set = []
    solution_accuracy = 0
    #make sure its not capturing the first column, this causes many issues 
    for i in range(1, data.shape[1]):
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for k in range(1, data.shape[1]):
            if not set(current_set_of_features).intersection({k}):
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                print('Using feature(s) ' + str(current_set_of_features + [k]) + ' accuracy is ' + str(accuracy * 100) + '%')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        if best_so_far_accuracy > solution_accuracy:
            solution_accuracy = best_so_far_accuracy
            solution_set.append(feature_to_add_at_this_level)
            print('Feature set ' + str(solution_set) + ' was best, accuracy is ' + str(solution_accuracy * 100) + '%')
        current_set_of_features.append(feature_to_add_at_this_level)
        return solution_set, solution_accuracy

def backward_elimination(data):
    current_set_of_features = []
    solution_set = []
    solution_accuracy = 0
    for k in range(1, data.shape[1]):
        current_set_of_features.append(k)
    for i in range(1, data.shape[1] - 1):
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0
        for k in current_set_of_features:
            removed_feature = current_set_of_features.copy()
            removed_feature.remove(k)
            accuracy = leave_one_out_cross_validation(data, removed_feature, -1)
            print('Removing ' + str(k) + ' in features ' + str(current_set_of_features) + ' accuracy is ' + str(accuracy * 100) + '%')
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                solution_set = current_set_of_features.copy()
                feature_to_remove_at_this_level = k
        if best_so_far_accuracy >= solution_accuracy:
            solution_accuracy = best_so_far_accuracy
            solution_set = current_set_of_features.copy()
        else: 
            pass 
        current_set_of_features.remove(feature_to_remove_at_this_level)
        return solution_set, accuracy


#run the main menu 
if __name__ == "__main__":
    main()