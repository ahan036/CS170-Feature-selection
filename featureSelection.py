import numpy as np 

#for running the dataset while testing just for my convenience
#https://www.geeksforgeeks.org/find-the-number-of-rows-and-columns-of-a-given-matrix-using-numpy/
'''def dataset(): 
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
    return np.shape(data)[0], np.shape(data)[1], data '''

#to choose the dataset and method 
#most of this is from the report guide, from the traceback code 
def main():
    print("Welcome to Ashley's Feature Selection Algorithm. \n")
    file_input = input("Type in the name of the file to test: ")
    print(file_input)
    print('\n')

    select_algo = int(input("Type the number of the algorithm you want to run. \n 1) Forward Selection \n 2) Backward Elimination \n"))
    #temp for easy testing 
    #instances, features, data = dataset()

    data = np.loadtxt(file_input)
    instances = np.shape(data)[0]
    features = np.shape(data)[1]

    print('This dataset has ' + str(features - 1) + ' features (not including the class attribute), with ' + str(instances) + ' instances.\n') 
    features = list(range(1, features))  
    all_features = leave_one_out_cross_validation(data, features, -1)
    print('Running nearest neighbor with all features, using \"leaving-one-out\" evalutation, I get an accuracy of ' + str(all_features) + '%')
    print('Beginning Search. \n')
    if select_algo == 1:
        subset, accuracy = forward_selection(data)
    else:
         subset, accuracy = backward_elimination(data)
    print('Finished search!! The best feature subset is ' + str(subset) +  ' which has an accuracy of ' + str(accuracy) + '%')

#pseudocode from the slides 
def leave_one_out_cross_validation(data, current_set, feature_to_add):

    #this is to make sure we add the new feature to our existing set, if its -1 we reached the first column
    if (feature_to_add != -1):
        #we need to copy or else we permanently alter current_set, bad practice 
        current_set = current_set.copy() 
        current_set.append(feature_to_add)
    
    number_correctly_classified = 0
    #shape tells us the number of rows in our data, basically we are looping for x # of rows 
    for i in range(data.shape[0]): 
        features = len(current_set)
        object_to_classify = data[i, current_set]
        label_object_to_classify = data[i][0]
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf

        for k in range(data.shape[0]):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k, current_set]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified / data.shape[0]
    #yay fixed my weird number issue 
    #https://www.geeksforgeeks.org/how-to-round-numbers-in-python/
    return round(accuracy * 100, 1)

#data = our file 
#current_set = the set of features we are selecting 
#feature_to_add = the feature we might add, we have to test the accuracy first 
#forward and backward selection from the slides : project_2 briefing
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
                print('Using feature(s) ' + str(current_set_of_features + [k]) + ' accuracy is ' + str(accuracy) + '%')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        if best_so_far_accuracy > solution_accuracy:
            solution_accuracy = best_so_far_accuracy
            solution_set.append(feature_to_add_at_this_level)
            print('Feature set ' + str(solution_set) + ' was best, accuracy is ' + str(solution_accuracy) + '%')
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
            print('Removing ' + str(k) + ' in features ' + str(current_set_of_features) + ' accuracy is ' + str(accuracy) + '%')

            #if current acc better than our best we change our local best, set the possible feature 
            #to remove as our k 
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                #solution_set = current_set_of_features.copy()
                feature_to_remove_at_this_level = k

        #if the accuracy is better than our solution acc, then we want the solution to stay the same 
        # remove the item from the current set aand save the set as our new solution set         
        if best_so_far_accuracy > solution_accuracy:
            solution_accuracy = best_so_far_accuracy
            current_set_of_features.remove(feature_to_remove_at_this_level)
            solution_set = current_set_of_features.copy()

        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(solution_accuracy) + '%')
    return solution_set, solution_accuracy


#run the main menu duhhh
if __name__ == "__main__":
    main()


#credit: 
#HEAVILYYYY referenced the project 2 briefing slides and the matlab code provided
# https://www.dropbox.com/scl/fo/blbkjaf1eyl94lij5wl2b/AAvNKn0YrnaX0oGPQn7ueFo?dl=0&e=1&preview=Project_2_Briefing.pptx&rlkey=alq2gb2ftsw73hcar4lk897r0
# i didnt know how to read in the columns and rows from the file so i used
#  https://www.geeksforgeeks.org/find-the-number-of-rows-and-columns-of-a-given-matrix-using-numpy/
# had the same issue with overly complicated decimal so i used rounding 
#https://www.geeksforgeeks.org/how-to-round-numbers-in-python/