from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Transforms the feature "sex" from string to the boolean feature "is_male"
def transform_sex(dataset):
    is_male = dataset["sex"].isin(["M"])
    sexColIndex = dataset.columns.get_loc("sex")
    dataset.insert(sexColIndex, "is_male", is_male, False)
    dataset.drop(["sex"], axis=1, inplace=True)


# Transforms the feature "blood_type" from string the boolean feature "SpecialProperty"
def transform_blood_type(dataset):
    bloodTypeColIndex = dataset.columns.get_loc("blood_type")
    specialProperty = dataset["blood_type"].isin(["O+", "B+"])
    dataset.insert(bloodTypeColIndex, "SpecialProperty", specialProperty, False)
    dataset.drop(["blood_type"], axis=1, inplace=True)


# Transforms the feature "current_location" from string the continuous features "latitude" & "longitude"
def split_location(dataset):
    location_series = dataset['current_location'].astype(str).str.split("', '")
    latitude_series = location_series.str[0].str[2:].astype(float)
    longitude_series = location_series.str[1].str[:-2].astype(float)
    locationColIndex = dataset.columns.get_loc("current_location")
    dataset.insert(locationColIndex, "latitude", latitude_series, False)
    dataset.insert(locationColIndex+1, "longitude", longitude_series, False)
    dataset.drop(["current_location"], axis=1, inplace=True)


# Transforms the feature "symptoms" from string separate boolean features
def splitSymptomsToSeparateColumns(dataset):
    sore_throat = dataset["symptoms"].str.contains("sore_throat", na=False)
    cough = dataset["symptoms"].str.contains("cough", na=False)
    shortness_of_breath = dataset["symptoms"].str.contains("shortness_of_breath", na=False)
    smell_loss = dataset["symptoms"].str.contains("smell_loss", na=False)
    fever = dataset["symptoms"].str.contains("fever", na=False)
    symptomsColIndex = dataset.columns.get_loc("symptoms")
    dataset.insert(symptomsColIndex, "sore_throat", sore_throat, False)
    dataset.insert(symptomsColIndex, "cough", cough, False)
    dataset.insert(symptomsColIndex, "shortness_of_breath", shortness_of_breath, False)
    dataset.insert(symptomsColIndex, "smell_loss", smell_loss, False)
    dataset.insert(symptomsColIndex, "fever", fever, False)
    dataset.drop(["symptoms"], axis=1, inplace=True)


# Transforms the feature "pcr_date" from string the ordinal feature "date_as_ordinal"
def pcr_date_toordinal(dataset):
    pcr_ordinal_date = [datetime.strptime(d, '%d-%m-%y').toordinal() for d in dataset["pcr_date"]]
    pd.Series(pcr_ordinal_date)
    pcrDateColIndex = dataset.columns.get_loc("pcr_date")
    dataset.insert(pcrDateColIndex + 1, "date_as_ordinal", pcr_ordinal_date, False)
    dataset.drop(["pcr_date"], axis=1, inplace=True)


# Converts all boolean features to int
def dataset_bool_to_int(dataset):
    for feature in dataset.columns.tolist():
        if dataset[feature].dtypes == "bool":
            dataset[feature] = dataset[feature].astype(int)


def prepare_data(training_data, new_data):
    # Creating copies of the datasets to be safe
    res = new_data.copy()
    model = training_data.copy()
    # Transforming all the features to numeric ones
    transform_sex(res)
    transform_sex(model)
    transform_blood_type(res)
    transform_blood_type(model)
    split_location(res)
    split_location(model)
    splitSymptomsToSeparateColumns(res)
    splitSymptomsToSeparateColumns(model)
    pcr_date_toordinal(res)
    pcr_date_toordinal(model)
    dataset_bool_to_int(res)
    dataset_bool_to_int(model)
    # Choosing which feature should be normalized in which scaler
    normalized = ["risk", "spread"]
    minMaxFeatures = ["patient_id", "date_as_ordinal", "PCR_01", "PCR_02", "PCR_03", "PCR_05", "PCR_06", "is_male",
                      "SpecialProperty", "fever", "smell_loss", "shortness_of_breath", "cough", "sore_throat"]
    standardFeatures = [x for x in model.columns if (x not in minMaxFeatures) and (x not in normalized)]
    # Setting up the 2 scalers
    minMaxScaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    standardScaler = StandardScaler(copy=False)
    # Fitting the scalers based on the model dataset
    minMaxScaler.fit(model[minMaxFeatures])
    standardScaler.fit(model[standardFeatures])
    # Applying the scalers to the result dataset
    res[minMaxFeatures] = minMaxScaler.transform(res[minMaxFeatures])
    res[standardFeatures] = standardScaler.transform(res[standardFeatures])
    return res
