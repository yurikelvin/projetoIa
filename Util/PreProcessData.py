import pandas as pd # Require the pandas module installed
import numpy as np

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

np.random.seed(10)

PERCENT_OF_DATA_TRAINING = 0.8

print 'Reading the csv'

df = pd.read_csv("../dataframes/true_car_listings.csv")

print 'Read completed...'
print 'Deleting the Vin unused column'

dfWithoutVin = df.drop('Vin', 1)
dfWithoutVin = dfWithoutVin.sample(5000)

# Create the dummy variable
dfWithoutVin = pd.get_dummies(dfWithoutVin)

sizeOfDf = len(dfWithoutVin.index)

sizeOfDfTraining = int(round(PERCENT_OF_DATA_TRAINING * sizeOfDf, 0))

print 'Sucessful remove the Vin column and calculate the number that represents 80% percent of data training of whole data'
print repr(sizeOfDfTraining) + ' represents (80%) of ' + repr(sizeOfDf)

indexesWholeData = dfWithoutVin.index.tolist()
indexesDataTraining = np.random.choice(dfWithoutVin.index, sizeOfDfTraining, replace=False)

print "Indexes of data training completed."

indexesDataTest = diff(indexesWholeData, indexesDataTraining) # The indexes of Data Test represents 20% of the whole of data

dataTraining = dfWithoutVin.drop(indexesDataTest)
dataTest = dfWithoutVin.drop(indexesDataTraining)

print 'Sucessful extract the data training and data test from whole data'

print 'Generating the output csv'

dataTraining.to_csv('../dataframes/true_car_listings_data_training.csv')
dataTest.to_csv('../dataframes/true_car_listings_data_test.csv')
