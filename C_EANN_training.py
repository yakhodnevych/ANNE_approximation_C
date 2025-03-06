import openpyxl
import numpy as np
from os.path import join, abspath
# Parallel training of an ensemble of artificial neural networks ANN_A, ANN_B1, ANN_B2,

# Definition of a function for computing the standard logistic activation function

def logistic(x):
    return 1.0 / (1 + np.exp(-x))


# Definition of a function for computing the derivative of the activation function

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


# network training parameters
epoch_count = 200  # training epochs of neural networks
alpha = 0.002 # the training speed coefficient value


def traininng_ANN(ANN_ID, epoch_count, alpha):
    # Training function of an artificial neural network with the identifier ANN_ID,
    # Reading network parameters and training examples from an MS Excel data file
    # with the sheet number (page) page_number

    page_number = 0  # Sheet number in the Excel data file
    # select the page number with training examples for the corresponding network - A or B1, or B2
    if ANN_ID == "A": page_number = 0

    if ANN_ID == "B1": page_number = 2

    if ANN_ID == "B2": page_number = 4

    # Open the specified file only to read the values (not read formulas),
    # it is created a full path to open the file based on the directory in which the program is running,
    # such full name is stored in data_path
    data_path = join('.', 'Data', "Training_Data.xlsx")
    data_path = abspath(data_path)

    Training_Data = openpyxl.open(data_path, read_only=True, data_only=True)
    # select the appropriate worksheet in the Excel workbook
    sheet_data = Training_Data.worksheets[page_number]

    # read directly from the file (number of input-output network pairs)
    max_row_data = sheet_data['A3'].value  # number of teaching examples

    # network parameters
    input_size = sheet_data['A5'].value  # number of inputs
    hidden_size = sheet_data['A7'].value  # number of neurons in the hidden layer
    output_size = sheet_data['A9'].value  # number of outputs

    print('The network parameters ANN_'+ANN_ID)
    print('number of inputs:', input_size)
    print('number of neurons in the hidden layer:', hidden_size)
    print('number of outputs:', output_size)
    print('number of teaching examples:', max_row_data)

    # Definition and initialization of matrices of the ANN input and output values
    len_ryadok = input_size

    characteristics_riverbed = np.zeros(
        shape=(max_row_data, len_ryadok))  # input - hydromorphological channel characteristics Q, B, h, Sf, n
    coef_C = np.zeros(shape=(max_row_data))  # output – the Chézy coefficient value C, or delta1, or delta2

    # The variable coef_C can take on different values depending on ANN_ID
    # in the case of ANN_A, the outputs of the network are the values of the coefficient C
    # in the case of ANN_B1, the outputs of the network are the refinement values delta1 for the values of the coefficient C
    # in the case of ANN_B2, the outputs of the network are the refinement values delta2 for the values of the coefficient C

    # filling the matrices for the network's input and output values with training data from the file

    for i in range(max_row_data):
        coef_C[i] = sheet_data[i + 2][input_size + 1].value
        for j in range(len_ryadok):
            characteristics_riverbed[i][j] = sheet_data[i + 2][j + 1].value


    # matrix of weight coefficients W_1 і W_2
    # are given by random values

    np.random.seed(1)
    W_1 = 0.02 * np.random.random((input_size, hidden_size)) - 0.01
    W_2 = 0.6 * np.random.random((hidden_size, output_size)) - 0.3

    # realization of training epochs
    # (alternate execution of the direct and reverse steps of the algorithm)
    # for all learning examples for 1 epoch

    correct = 0  # number of satisfactory learning outcomes
    total = 0  # number of studied learning examples
    error_delta = 0.001  # the network error value

    for iteration in range(epoch_count):
        layer_2_e2 = 0  # the sum of squares deviations
        for i in range(len(characteristics_riverbed)):
            # direct course of calculations
            layer_0 = characteristics_riverbed[i:i + 1]
            layer_1 = logistic(np.dot(layer_0, W_1))
            layer_2 = np.dot(layer_1, W_2)

            layer_2_e2 += np.sum((layer_2 - coef_C[i:i + 1]) ** 2)

            # reverse course of calculations
            layer_2_delta = (coef_C[i:i + 1] - layer_2)
            layer_1_delta = layer_2_delta.dot(W_2.T) * logistic_deriv(layer_1)

            W_2 = W_2 + alpha * layer_1.T.dot(layer_2_delta)
            W_1 = W_1 + alpha * layer_0.T.dot(layer_1_delta)

            # training accuracy assessment
            if (np.abs(layer_2_delta) < error_delta):
                correct += 1
            total += 1
            if ((iteration == epoch_count - 1) and (i % 10 == 9)):
                print("Iteration:" + str(iteration) + ", step:" + str(i))
                print("calculated output = " + str(layer_2))
                print("reference output = " + str(coef_C[i]))
                print("Quadratic Error:" + str(layer_2_e2))
                print("Training Accuracy:" + str(correct * 100 / float(total)))  # точність навчання

    # checking the accuracy (adequacy) of forecasting
    # reading and downloading a test sample of data

    # select the next worksheet with test examples in the MS Excel file book
    sheet_data = Training_Data.worksheets[int(page_number+1)]

    # the number of test examples is read directly from the file
    max_row_test = sheet_data['A3'].value

    characteristics_riverbed_test = np.zeros(
        shape=(max_row_test, len_ryadok))  # input - test values of hydromorphological channel characteristics Q, B, h, Sf, n
    coef_C_test = np.zeros(shape=(max_row_test))  # output - the Chézy coefficient test values С, or delta 1, or delta2

    # filling the matrices for the network's input and output values
    # with test examples
    for i in range(max_row_test):
        coef_C_test[i] = sheet_data[i + 2][input_size + 1].value
        for j in range(len_ryadok):
            characteristics_riverbed_test[i][j] = sheet_data[i + 2][j + 1].value

    correct = 0  # number of satisfactory forecasting results
    total = 0  # number of checked test examples

    # calculation of network outputs and accuracy assessment
    for i in range(max_row_test):
        # direct course of calculations
        layer_0 = characteristics_riverbed_test[i:i + 1]
        layer_1 = logistic(np.dot(layer_0, W_1))
        layer_2 = np.dot(layer_1, W_2)

        # forecasting accuracy assessment
        if (np.abs(coef_C_test[i:i + 1] - layer_2) < error_delta):
            correct += 1
        total += 1
        print('example ', i, ', test output: ', coef_C_test[i], ', calculated output: ', layer_2)
    print('all test examples: ', total)
    print("Test Accuracy:" + str(correct * 100 / float(total)))  # forecasting accuracy on test examples

    Training_Data.close()  # closing of the MS Excel data file

    # entry in the text file of learning outcomes - matrices of weights W_1 та W_2 with the identifier ANN_ID,
    # if the file contains information, data will be deleted

    # it is created a full path to open the file based on the directory in which the program is running,
    # such full name is stored in data_path
    data_path = join('.', 'Data', "weights_matrix_1_"+ANN_ID+".txt")
    data_path = abspath(data_path)

    f = open(data_path, 'w')
    for i in range(0, input_size - 1):
        for j in range(0, hidden_size - 1):
            f.write(str(W_1[i][j]) + ' ')  # writing to the file only the values of the array through space, others are not transmitted
        f.write(str(W_1[i][hidden_size - 1]) + '\n')
    for j in range(0, hidden_size - 1):
        f.write(str(W_1[input_size - 1][j]) + ' ')
    f.write(str(W_1[input_size - 1][
                    hidden_size - 1]))
    f.close()

    # it is created a full path to open the file based on the directory in which the program is running,
    # such full name is stored in data_path
    data_path = join('.', 'Data', "weights_matrix_2_"+ANN_ID+".txt")
    data_path = abspath(data_path)

    f = open(data_path, 'w')
    for i in range(0, hidden_size - 1):
        for j in range(0, 1):
            f.write(str(W_2[i][j]) + '\n')
    f.write(str(W_2[hidden_size - 1][
                    0]))
    f.close()

    print('For ANN_'+ANN_ID+' the adjusted matrices of weight coefficients successfully saved.')
 # ***************** end of function traininng_ANN *****************

# Training of an ensemble of artificial neural networks ANN_A, ANN_B1, ANN_B2,
# training an artificial neural network with the identifier ANN_ID, the number of training epochs, the learning rate coefficient alpha
traininng_ANN("A", 100, alpha) # ANN_A
traininng_ANN("B1", 600, alpha) # ANN_B1
traininng_ANN("B2", 600, alpha) # ANN_B2

print('Press any key to end the program.')
input()
