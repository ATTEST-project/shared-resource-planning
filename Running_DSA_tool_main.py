# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
#ATTEST - WP4- Task 4.6 - On-line dynamic security assessment tool
# Created By  : Pedro Barbeiro pedro.p.barbeiro@inesctec.pt
# Created Date: 11/03/2022
# Original version ='1.0'
# Current version ='1.1'
#Last modifications 17/04/2023
# -----------------
import os
import csv
import math
import copy
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow import keras
import time


def calc_rmse(y_pred, y_true):

    mse_rocof = 0
    mse_freq = 0

    for i in range(len(y_pred)):
        mse_rocof += (y_true[i][0] - y_pred[i][0]) ** 2
        mse_freq += (y_true[i][1] - y_pred[i][1]) ** 2

    mse_rocof /= len(y_pred)
    mse_freq /= len(y_pred)

    rmse_rocof = math.sqrt(mse_rocof)
    rmse_freq = math.sqrt(mse_freq)

    return rmse_rocof, rmse_freq, mse_rocof, mse_freq

def calc_rmse_1(y_pred, y_true):

    mse_rocof = 0
    mse_freq = 0

    for i in range(len(y_pred)):
        mse_rocof += ((y_true[i][0] - y_pred[i][0])/50) ** 2
        mse_freq += ((y_true[i][1] - y_pred[i][1])/50) ** 2

    mse_rocof /= len(y_pred)
    mse_freq /= len(y_pred)

    rmse_rocof = math.sqrt(mse_rocof)
    rmse_freq = math.sqrt(mse_freq)

    return rmse_rocof, rmse_freq, mse_rocof, mse_freq

def convert_pu(y_pred, y_true):

    nadir_pred = []
    rocof_pred = []
    nadir_true = []
    rocof_true = []

    for i in range(len(y_pred)):
        nadir_pred.append(y_pred[i][1]/50)
        rocof_pred.append(y_pred[i][0]/50)
        nadir_true.append(y_true[i][1]/50)
        rocof_true.append(y_true[i][0]/50)

    return nadir_pred, rocof_pred, nadir_true, rocof_true

def convert_data(data):

    list_input = []
    list_output = []
    for i in range(len(data)):
        aux1 = [data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]]
        list_input.append(aux1)

        aux2 = [data[i][6], data[i][7]]
        list_output.append(aux2)

    return np.asarray(list_input), np.asarray(list_output)


def define_trainning_set_limits(training_inputs, training_outputs):
    '''

    @param training_inputs: trainning set inputs
    @param training_outputs: trainning set outputs
    @return: returns an array containing the max and min value of each trainning input and output
    '''

    thermal_power = [min(training_inputs[:, 0]), max(training_inputs[:, 0])]
    pv_power = [min(training_inputs[:, 1]), max(training_inputs[:, 1])]
    wind_power = [min(training_inputs[:, 2]), max(training_inputs[:, 2])]
    spinning_reserve = [min(training_inputs[:, 3]), max(training_inputs[:, 3])]
    sm_h = [min(training_inputs[:, 4]), max(training_inputs[:, 4])]
    sc_h = [min(training_inputs[:, 5]), max(training_inputs[:, 5])]

    rocof = [min(training_outputs[:, 0]), max(training_outputs[:, 0])]
    freq_nadir = [min(training_outputs[:, 1]), max(training_outputs[:, 1])]

    inputs_limits = [thermal_power, pv_power, wind_power, spinning_reserve, sm_h, sc_h]
    output_limits = [rocof, freq_nadir]

    return inputs_limits, output_limits


def import_created_scenarios(csv_file, rocof_position, disturbance):
    '''

    @param csv_file: list of lists: [thermal power, pv power, wind power, spinning reserve, sm inertia, sc inertia
    rocof, freq nadir]
    @return:
    '''

    # open csv file
    path = os.path.dirname(os.path.realpath(__file__))
    ifile = open(os.path.join(path, csv_file), "r")

    # read data from csv file
    read = csv.reader(ifile)

    list_input = []
    output_rocof = []
    output_fnadir = []
    data = []
    Rocof_true = 9999 # Comment to test predictions against true values (change code below to the 1st version)
    Nadir_true = 9999 # Comment to test predictions against true values (change code below to the 1st version)

    i = 0
    del_item = []

    firstline = True
    nr_OPs = 0


    for row in read:
        if firstline:  # skip first line
            firstline = False
            continue

        if row[0] == "/…" or nr_OPs == Nr_of_OPs_to_be_tested:  # Stop reading and Exit
            break
        # if (float(row[7]) > 47) and (float(row[7]) < 51) and (float(row[6]) < 6): #PNPB comentei apr aler todos casos isto era que Gouveia fazia para desprezar casos que eram muito criticos
        # algo que para mim nao faz sentido ver com Gouveia e Carlos
        if disturbance == 2:
            data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[20]),
                         float(row[rocof_position]), float(row[7])])
            aux = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[20])]
        else: #Disturbance = 1, etc.
            data.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
                         # 6 Vars Input e 2 Output
                         float(Rocof_true), float(Nadir_true)]) #Replace by float(row[rocof_position]), float(row[7])]) to  o test predictions against true values
            aux = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])]
        list_input.append(aux)

        output_rocof.append(float(Rocof_true))
        output_fnadir.append(float(Nadir_true))
        nr_OPs += 1

    list_output = [output_rocof, output_fnadir]


    return np.asarray(list_input), np.asarray(list_output).T, np.asarray(data)

def import_config_file_data(csv_file, disturbance):

    # open csv file
    path = os.path.dirname(os.path.realpath(__file__))
    ifile = open(os.path.join(path, csv_file), "r")

    # read data from csv file
    read = csv.reader(ifile)

    inputs_limits = []
    output_limits = []
    Nr_of_ANNs = 0
    Nr_of_Input_Vars = 0
    Nr_of_Output_Vars = 0
    counter_rows = 0

    for row in read:

        if counter_rows == 0 or counter_rows == 2 or counter_rows == 3 or counter_rows == 5 or counter_rows == 6 or counter_rows == 8 or counter_rows == 9 or counter_rows == 11 or counter_rows == 12 or counter_rows == 13 or counter_rows == 14:  # skips specific lines
            counter_rows += 1
            continue
        if counter_rows == 1:
            Nr_of_OPs_to_be_tested = int(row[0])
        if counter_rows == 4:
            Nr_of_ANNs = int(row[0])
        if counter_rows == 7:
            Nr_of_Input_Vars = int(row[0])
        if counter_rows == 10:
            Nr_of_Output_Vars = int(row[0])

        if counter_rows == 15 + (Nr_of_ANNs*(Nr_of_Input_Vars + Nr_of_Output_Vars)):  # pos. 15 is where stats the ANNs max and min vars inputdata skip first line
            break

        if counter_rows >= 15 and counter_rows <= 22: #First ANN
            inputs_limits.append([float(row[3]), float(row[4])])

        if counter_rows >= 21 and counter_rows <= 22:  # First ANN
            output_limits.append([float(row[3]), float(row[4])])
        counter_rows += 1

    return Nr_of_OPs_to_be_tested, Nr_of_ANNs, Nr_of_Input_Vars, Nr_of_Output_Vars, np.asarray(inputs_limits), np.asarray(output_limits)

def normalization(training_inputs, training_outputs, flag, inputs_limits, output_limits):
    '''

    @param training_inputs: Array or matrix containing inputs
    @param training_outputs: Array or matrix containing outputs
    @param flag: 1 -> normalize input/output to value between 0-1;
           flag: 2 -> transform normalized value to the original value;
    @return:
    '''

    if flag == 1:
        for i in range(len(training_inputs)):
            # print("Training set before: {}".format(training_inputs[i]))
            for pos in range(training_inputs.shape[1]):
                #if pos == 5:
                #    training_inputs[i][pos] = 0
                #else:
                training_inputs[i][pos] = (training_inputs[i][pos] - inputs_limits[pos][0]) / (inputs_limits[pos][1] - inputs_limits[pos][0])
                if training_inputs[i][pos] == 'nan':
                    training_inputs[i][pos] = 0
            # print("Training set after: {}".format(training_inputs[i]))

        for i in range(len(training_outputs)):
            # print("Training set before: {}".format(training_outputs[i]))
            for pos in range(training_outputs.shape[1]):
                training_outputs[i][pos] = (training_outputs[i][pos] - output_limits[pos][0]) / (output_limits[pos][1] - output_limits[pos][0])
            # print("Training set after: {}".format(training_outputs[i]))

    if flag == 2:
        for i in range(len(training_inputs)):
            for pos in range(training_inputs.shape[1]):
                training_inputs[i][pos] = training_inputs[i][pos] * (inputs_limits[pos][1] - inputs_limits[pos][0]) + inputs_limits[pos][0]
        for i in range(len(training_outputs)):
            for pos in range(training_outputs.shape[1]):
                training_outputs[i][pos] = training_outputs[i][pos] * (output_limits[pos][1] - output_limits[pos][0]) + output_limits[pos][0]

    if flag == 3:
        for pos in range(len(training_inputs)):
            training_inputs[pos] = training_inputs[pos] * (inputs_limits[pos][1] - inputs_limits[pos][0]) + inputs_limits[pos][0]
        for pos in range(len(training_outputs)):
            training_outputs[pos] = training_outputs[pos] * (output_limits[pos][1] - output_limits[pos][0]) + output_limits[pos][0]

    return training_inputs, training_outputs


def write_training_limits(inputs_limits, output_limits, output_file):

    file_name = "training_limits_" + output_file + '.csv'

    with open(file_name, 'w') as f1:
        f1.write("Input limits:\n")
        for i in range(len(inputs_limits)):
            f1.write("%s, %s\n" % (inputs_limits[i][0], inputs_limits[i][1]))
        f1.write("Output limits:\n")
        for i in range(len(output_limits)):
            f1.write("%s, %s\n" % (output_limits[i][0], output_limits[i][1]))


def write_results(file_name, predictions_converted, initial_state_security, SC_Needed, predictions_converted_new_secure_system):

    # with open(file_name, 'w') as f:
    #     for i in range(predictions_converted.shape[0]):
    #
    #         f.write("%s, %s, %s, %s, %s\n" % (i + 1, test_outputs_original[i][0], test_outputs_original[i][1],
    #                                     predictions_converted[i][0], predictions_converted[i][1]))

    with open(file_name, 'w') as f:
        #f.write("ANN / Contingency / Disturbance Id, Estimated RoCoF (Hz/s), Estimated nadir (deviation to the Nominal Freq.) (Hz), OP Secure?, SC(s) needed to ensure security for one contingency (see Id), Estimated RoCoF (Hz/s) with SCs, Estimated nadir with SC(s) (deviation to the Nominal Freq.) (Hz)\n")
        for i in range(Nr_of_OPs_to_be_tested):
            f.write("OP Id, Estimated RoCoF (Hz/s), Estimated nadir (deviation to the Nominal Freq.) (Hz), OP Secure?, SC(s) needed to ensure security for one contingency (see Id), Estimated RoCoF (Hz/s) with SCs, Estimated nadir with SC(s) (deviation to the Nominal Freq.) (Hz)\n")
            f.write("%s, %s, %s, %s, %s, %s, %s\n" % (i + 1, predictions_converted[i][0], predictions_converted[i][1],
                                              initial_state_security[i], SC_Needed[i], predictions_converted_new_secure_system[i][0], predictions_converted_new_secure_system[i][1]))
            f.write("SC(s) needed to ensure system security for all contingencies /disturbances\n")
            f.write("%s\n\n" %SC_Needed[i])

if __name__ == "__main__":

    print("Execution Process of the DSA tool has started")

    start = time.time()

    disturbance = 1 #PNPB - Bus Short circuits

    if disturbance == 1:
        ANN_Trained_File = 'ANN_1.h5'
        input_test = 'test_cases.csv'
        output_file = 'ANN_Results.csv'
        config_file = 'config.csv'

    #rocof_min = 12
    #rocof_max = 11
    #rocof_abs = 6

    rocof_position = 6 #PNPB - Alteria, é a posiçao do ROCOF do test_set a começar em 0

    Nr_of_OPs_to_be_tested, Nr_of_ANNs, Nr_of_Input_Vars, Nr_of_Output_Vars, inputs_limits, output_limits = import_config_file_data(config_file, disturbance)

    test_inputs_original, test_outputs_original, data2 = import_created_scenarios(input_test, rocof_position, disturbance)

    test_inputs = copy.deepcopy(test_inputs_original)
    test_outputs = copy.deepcopy(test_outputs_original)

    # normalizing testing data set
    test_inputs_normalized, test_outputs_normalized = normalization(test_inputs, test_outputs, 1, inputs_limits, output_limits)

    model = keras.models.load_model(ANN_Trained_File)
    predictions = model.predict(test_inputs_normalized) #Run ANN

    # convert predictions results
    [], predictions_converted = normalization([], predictions, 2, inputs_limits, output_limits)
    SC_Needed = []
    initial_state_security =[]
    test_inputs = copy.deepcopy(test_inputs_original)
    #Check if the system is secure or unsecure for each OP and calculate the SCs required to ensure security for unsecure states
    for i in range(Nr_of_OPs_to_be_tested):
        if predictions_converted[i][0] >= 2.5 or predictions_converted[i][1] >= 2.0:
            test_inputs[i][2] = 866.4
            test_inputs[i][3] = 1619.8
            test_inputs[i][4] = 2300.0
            SC_Needed.append("SC1, SC2, SC3")
            initial_state_security.append("No") #0--> unsecure; 1--> secure
        elif (predictions_converted[i][0] >= 2.2 and predictions_converted[i][0] < 2.5) or (predictions_converted[i][1] >= 1.8 and predictions_converted[i][1]< 2.0):
            test_inputs[i][3] = 1619.8
            test_inputs[i][4] = 2300.0
            SC_Needed.append("SC2, SC3")
            initial_state_security.append("No")
        elif (predictions_converted[i][0] >= 2.0 and predictions_converted[i][0] < 2.2) or (predictions_converted[i][1] >= 1.6 and predictions_converted[i][1]< 1.8):
            test_inputs[i][2] = 866.4
            test_inputs[i][4] = 2300.0
            SC_Needed.append("SC1, SC3")
            initial_state_security.append("No")
        elif (predictions_converted[i][0] >= 1.8 and predictions_converted[i][0] < 2.0) or (predictions_converted[i][1] >= 1.4 and predictions_converted[i][1] < 1.6):
            test_inputs[i][2] = 866.4
            test_inputs[i][3] = 1619.8
            SC_Needed.append("SC1, SC2")
            initial_state_security.append("No")
        elif (predictions_converted[i][0] >= 1.5 and predictions_converted[i][0] < 1.8) or (predictions_converted[i][1] >= 1.2 and predictions_converted[i][1] < 1.4):
            test_inputs[i][4] = 2300.0
            SC_Needed.append("SC3_bus38")
            initial_state_security.append("No")
        elif (predictions_converted[i][0] >= 1.2 and predictions_converted[i][0] < 1.5) or (predictions_converted[i][1] >= 1.0 and predictions_converted[i][1] < 1.2):
            test_inputs[i][3] = 1619.8
            SC_Needed.append("SC2")
            initial_state_security.append("No")
        elif (predictions_converted[i][0] >= 1.0 and predictions_converted[i][0] < 1.2) or (predictions_converted[i][1] >= 0.8 and predictions_converted[i][1] < 1.0):
            test_inputs[i][2] = 866.4
            SC_Needed.append("SC1")
            initial_state_security.append("No")
        else: #RocoF < 1 Hz/s and nadir <0.8 Hz
            initial_state_security.append("Yes") #Secure
            SC_Needed.append("Not needed")

    # normalizing testing data set
    test_inputs_normalized, test_outputs_normalized = normalization(test_inputs, test_outputs, 1, inputs_limits, output_limits)
    model = keras.models.load_model(ANN_Trained_File)
    predictions_new_secure_system = model.predict(test_inputs_normalized)  # Run ANN
    # convert predictions results
    [], predictions_converted_new_secure_system = normalization([], predictions_new_secure_system, 2, inputs_limits, output_limits)

    # create csv file with results
    write_results(output_file, predictions_converted, initial_state_security, SC_Needed, predictions_converted_new_secure_system)

    x = "results.csv"
    y = "No errors occurred: The tool was run successfully for %s ANN (%s Contingency) and %s OPs" %(repr(Nr_of_ANNs), repr(Nr_of_ANNs), repr(Nr_of_OPs_to_be_tested))
    print("Reading and Loading Input Files")
    print("Running %i ANN model to perform DSA for %i OPs" %(Nr_of_ANNs,Nr_of_OPs_to_be_tested))
    print("Synchronizing data and compiling results")
    print("Process finished with exit Code 0 - %s" %repr(y))
    print("Please check %s file" %repr(x))
    end = time.time()
    elapsed_time = end - start
    print("Elapsed Time: %.3f seconds" %elapsed_time)