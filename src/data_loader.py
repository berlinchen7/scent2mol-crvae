"""
This file is to encode SMILES and SELFIES into one-hot encodings
"""

import numpy as np
import pandas as pd

import selfies as sf
import pyrfume


def smile_to_hot(smile, largest_smile_len, alphabet):
    """Go from a single smile string to a one-hot encoding.
    """

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded), char_to_int


def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """Convert a list of smile strings to a one-hot encoding

    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """

    hot_list = []
    for s in smiles_list:
        _, onehot_encoded, char_to_int = smile_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list), char_to_int


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)

def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

def multiple_hot_to_smiles(onehot_encoded: np.array, int_to_char: dict) -> list:
    """
    Params:
        onehot_encoded: has shape [# of molecules, max molecule size, alphabet size]
        int_to_char: mapping from one hot index to smiles char
    Return:
        list of smiles notation as type str
    """
    smiles_ret = []
    for i in range(onehot_encoded.shape[0]):
        smiles_ret.append(hot_to_smiles(onehot_encoded[i, :, :], int_to_char))
    return smiles_ret

def hot_to_smiles(onehot_encoded: np.array, int_to_char: dict) -> str:
    """
    Params:
        onehot_encoded: has shape [max molecule size, alphabet size]
        int_to_char: mapping from one hot index to smiles char
    """
    index_encoded = np.argmax(onehot_encoded, axis=1)
    char_list = [int_to_char[ind] for ind in index_encoded]
    smiles_str = ''.join(char_list).strip() # .strip() gets rid of the leading and trailing white spaces
    return smiles_str


def multiple_hot_to_selfies(onehot_encoded: np.array, int_to_char: dict) -> list:
    """
    Params:
        onehot_encoded: has shape [# of molecules, max molecule size, alphabet size]
        int_to_char: mapping from one hot index to selfies string
    Return:
        list of selfies notation as type str
    """
    selfies_ret = []
    for i in range(onehot_encoded.shape[0]):
        selfies_ret.append(hot_to_selfies(onehot_encoded[i, :, :], int_to_char))
    return selfies_ret

def hot_to_selfies(onehot_encoded: np.array, int_to_char: dict) -> str:
    """
    Params:
        onehot_encoded: has shape [max molecule size, alphabet size]
        int_to_char: mapping from one hot index to selfies string
    """
    index_encoded = np.argmax(onehot_encoded, axis=1)
    char_list = [int_to_char[ind] for ind in index_encoded]
    selfies_str = ''.join(char_list).strip() # .strip() gets rid of the leading and trailing white spaces
    return selfies_str


def generate_scent_labels(dataset: str) -> list:
    # Use train-test split given in Leffingwell Dataset - except add in validation set (have 70% train, 10% validation, 20% test rather than 80% train & 20% test)
    # Load data
    scentdata = pyrfume.load_data("leffingwell/leffingwell_data.csv", remote=True)

    # Code used to create train, test & validation sets (based on the splits given in Leffingwell Dataset)
    testData = scentdata[scentdata["labels_train/test"] == 0]
    numTestData = len(testData)

    trainAndValidationData = scentdata[scentdata["labels_train/test"] == 1]

    numTrainAndValidationData = len(trainAndValidationData)
    trainAndValidationData = trainAndValidationData.reset_index()

    numMoleculesInDataset = numTestData + numTrainAndValidationData

    # randomly select indices from trainAndValidation data - validation set = 10% of entire dataset
    numValidationData = int(0.10 * numMoleculesInDataset)
    validationIndices = np.random.choice(
        a=numTrainAndValidationData, size=numValidationData, replace=False
    )  # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    validationData = trainAndValidationData.iloc[
        validationIndices
    ]  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
    trainData = trainAndValidationData.drop(
        index=validationIndices
    )  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html


    # Use smaller set of data (just to check code does not crash)
    # NOTE: comment out the next 3 lines when actually training/evaluating model to use the entire dataset
    trainData = trainData.sample(frac=0.01, random_state=0).reset_index(drop=True)
    validationData = validationData.sample(frac=0.01, random_state=0).reset_index(drop=True)
    testData = testData.sample(frac=0.8, random_state=0).reset_index(
        drop=True
    )  # Sample more from test set to avoid error that there is only 1 molecule with a certain scent when calculating AUROC score

    # Code to generate list of all scent labels (scentClasses)
    numMolecules = len(scentdata.odor_labels_filtered)
    numClasses = 112  # No odorless class
    scentClasses = pd.read_csv("src/molecular_ae/scentClasses.csv")
    scentClasses = scentClasses["Scent"].tolist()
    moleculeScentList = []
    for i in range(numMolecules):
        scentString = scentdata.odor_labels_filtered[i]
        temp = scentString.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace("'", "")
        temp = temp.replace(" ", "")
        scentList = temp.split(",")
        if "odorless" in scentList:
            scentList.remove("odorless")
        moleculeScentList.append(scentList)

    # Generate moleculeScentList_train, moleculeScentList_test, moleculeScentList_validation
    numTrainMolecules = len(trainData.odor_labels_filtered)
    moleculeScentList_train = []
    for i in range(numTrainMolecules):
        scentString = trainData.odor_labels_filtered[i]
        temp = scentString.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace("'", "")
        temp = temp.replace(" ", "")
        scentList = temp.split(",")
        if "odorless" in scentList:
            scentList.remove("odorless")
        moleculeScentList_train.append(scentList)

    numValidationMolecules = len(validationData.odor_labels_filtered)
    moleculeScentList_validation = []
    for i in range(numValidationMolecules):
        scentString = validationData.odor_labels_filtered[i]
        temp = scentString.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace("'", "")
        temp = temp.replace(" ", "")
        scentList = temp.split(",")
        if "odorless" in scentList:
            scentList.remove("odorless")
        moleculeScentList_validation.append(scentList)

    numTestMolecules = len(testData.odor_labels_filtered)
    moleculeScentList_test = []
    for i in range(numTestMolecules):
        scentString = testData.odor_labels_filtered[i]
        temp = scentString.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace("'", "")
        temp = temp.replace(" ", "")
        scentList = temp.split(",")
        if "odorless" in scentList:
            scentList.remove("odorless")
        moleculeScentList_test.append(scentList)
    # print(moleculeScentList_test)

    return trainData, moleculeScentList_train, validationData, moleculeScentList_validation, testData, moleculeScentList_test