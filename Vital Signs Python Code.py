import random as rand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


pd.set_option('display.width', None)

# Phase 1:

# set parameters for vital sign variables, L(min) and H(max)range:
start = 101
tempL, tempH = 36, 39
hRateL, hRateH = 55, 100
pulseL, pulseH = 55, 100
bpSeq = (120, 121)
respRateL, respRateH = 11, 17
oxySatL, oxySatH = 93, 100
phL, phH = 7.1, 7.6

# Set variables containing parameters for abnormal rates:
abnTemp = [36, 39]
abnHRate = [55, 56, 57, 58, 59, 100]
abnPulse = [55, 56, 57, 58, 59, 100]
abnBp = [121]
abnRespRate = [11, 17]
abnOxySat = [93, 94]
abnPh = [7.1, 7.2, 7.6]

n = 1000

# create function to generate data frame from random data:
def myHealthCare(n):
    # create the vital signs data:
    rand.seed(109)
    tStamp = [i for i in range(start, start + n, 1)]
    temp = [rand.randint(tempL, tempH) for _ in range(n)]
    hRate = [rand.randint(hRateL, hRateH) for _ in range(n)]
    pulse = [rand.randint(pulseL, pulseH) for _ in range(n)]
    bp = [rand.choice(bpSeq) for _ in range(n)]
    respRate = [rand.randint(respRateL, respRateH) for _ in range(n)]
    oxySat = [rand.randint(oxySatL, oxySatH) for _ in range(n)]
    ph = [round(rand.uniform(phL, phH), 1) for _ in range(n)]

    # create data frame from dictionary
    data = {"Timestamp": tStamp,
            "Temperature": temp,
            "Heart Rate": hRate,
            "Pulse": pulse,
            "Blood Pressure": bp,
            "Respiratory Rate": respRate,
            "Oxygen Saturation": oxySat,
            "pH": ph}
    myHealthDf = pd.DataFrame(data)
    # set index to be the same as the time stamp value
    recordIndex = pd.Index([i for i in range(101, 101 + n, 1)])
    myHealthDf = myHealthDf.set_index(recordIndex)
    return myHealthDf

print(myHealthCare(n).head())

print()
print("-----------------------------------------------")
print()

# Phase 2:
#create small sample:
np.random.seed(109)
smallSampDF = pd.DataFrame(myHealthCare(n).sample(frac=0.05, random_state=109))

abnMeasure = abnPulse
abnColumn = "Pulse"

#a) abnormal sign analytics function for Pulse:
def abnormalSignAnalytics(abnMeasure, abnColumn):
    #create new DF from sampleDF containing only required abnormal values for the selected abnormal measure
    abnormalDF = pd.DataFrame(smallSampDF.iloc[np.where(smallSampDF[abnColumn].isin(abnMeasure))])
    # edit DF to only include 2 columns required.
    abnormalDF = abnormalDF[["Timestamp", abnColumn]]
    #create  a list of the values in the DF
    lst = []
    for index, row in abnormalDF.iterrows():
        lst.append((row["Timestamp"], row["Pulse"]))
    return lst

print("Abnormal sign values for",str(abnColumn),
      "counted" , len(abnormalSignAnalytics(abnMeasure, abnColumn)),
      "records;",abnormalSignAnalytics(abnMeasure, abnColumn))

print()
print("-----------------------------------------------")
print()
# b) Present a frequency histogram of Pulse rates from smallSampleDF
freqMeasure = "Pulse"

def frequencyAnalytics(freqMeasure):
    signFreq = pd.DataFrame(smallSampDF)
    maxf = max(signFreq[freqMeasure])
    minf = min(signFreq[freqMeasure])
    bins = maxf-minf
    plot = plt.hist(signFreq[freqMeasure], color="green", bins=bins, alpha = 0.5)
    plt.xlim(minf-2, maxf+2)
    plt.title("Histogram: Frequency Analytics")
    plt.xlabel(freqMeasure)
    plt.ylabel("Frequency")

    return plot

frequencyAnalytics(freqMeasure), plt.show()

#c) plot for a) abnormalSignAnalytics

numlist = []
for num in (abnormalSignAnalytics(abnMeasure,abnColumn)):
     numlist.append(num[1])
     numlist.sort()
print(numlist)

bins = max(numlist)-min(numlist)
plt.hist(numlist, bins=bins)
plt.xlabel(abnColumn)
plt.ylabel("Frequency")
plt.title("Abnormal Sign Analytics Histogram")
plt.show()
plt.close()
print()
print("-----------------------------------------------")
print()


# Phase 3
# health analyser function - a query mechanism to search for
# a given sign value. e.g pulse, 56. Which returns the list with all
# other associated records

# Using the data frame & a boolean value to search for specified value for pulse.
value = 56

def healthAnalyser1(value):
    df = myHealthCare(n)
    # Create new df; filter using a boolean:
    dfAnalyser1 = pd.DataFrame(df[df.Pulse == value])
    lstVals = dfAnalyser1.values.tolist()
    return lstVals

print(healthAnalyser1(value))
print("Number of records: " + str(len(healthAnalyser1(value))))

print()

# linear search
def healthAnalyser2(value):
    df = myHealthCare(n)
    dflst2 = df.values.tolist()
    lstVals2 = []
    for row in range(len(dflst2)):
        check = dflst2[row][3]
        if check == value:
            lstVals2.append(dflst2[row])

    return lstVals2

print(healthAnalyser2(value))
print("Number of records: " + str(len(healthAnalyser2(value))))

print()


# in order to search, sort the data first using merge sort.
# sort based on the index position of the sign of interest, pulse is [3].
# ideally interpolation search would be used following. In the code below,
# a linear search is used (i couldn't get interpolation search to work for an array :(  ).
# healthAnalyser3 calls mergeSort, which calls merge, then calls search.

def healthAnalyser3():
    df = myHealthCare(n)
    dflst3 =df.values.tolist()
    dflst3Sorted = mergeSort(dflst3)
    result = search(dflst3Sorted)
    return result

def mergeSort(lst):
    if len(lst) <= 1:
        return lst

    middleIndex = len(lst) // 2
    leftSplit = lst[:middleIndex]
    rightSplit = lst[middleIndex:]

    leftSorted = mergeSort(leftSplit)
    rightSorted = mergeSort(rightSplit)

    return merge(leftSorted, rightSorted)


def merge(left, right):
    result = []

    while (left and right):
        if left[0][3] < right[0][3]:
            result.append(left[0])
            left.pop(0)
        else:
            result.append(right[0])
            right.pop(0)

    if left:
        result += left
    if right:
        result += right

    return result

def search(lst):
    lstVals =[]
    for row in lst:
        if row[3] == value:
            lstVals.append(row)
        if row [3] > value:
            break

    return lstVals



print(healthAnalyser3())
print("Number of records: " + str(len(healthAnalyser3())))

print()
print("-----------------------------------------------")

starttime = time.time()
healthAnalyser1(value)
print("Time for healthAnalyser1: ", round((time.time() - starttime), 2), "seconds")
starttime = time.time()
healthAnalyser2(value)
print("Time for healthAnalyser2: ", round((time.time() - starttime), 2), "seconds")
starttime = time.time()
healthAnalyser3()
print("Time for healthAnalyser3: ", round((time.time() - starttime), 2), "seconds")

print()
print("-----------------------------------------------")

#plot heart rate values for records having pulse rate 56:

heartRateVals = []
for num in (healthAnalyser1(value)):
     heartRateVals.append(num[2])
     heartRateVals.sort()

bins = int(max(heartRateVals))-int(min(heartRateVals))
plt.hist(heartRateVals, bins=bins, color="pink")
plt.xlabel("Heart Rates")
plt.ylabel("Frequency")
plt.title("Heart Rate Values Where Pulse is 56")
plt.show()

print()
print("-----------------------------------------------")
print()

plt.close()

#Phase 4:

#just a reminder of the functions:
# myHealthCare(n)
# abnormalSignAnalytics(n)
# frequencyAnalytics(n)
# healthAnalyser1(n)

def benchmarkingmyHealthcare(n):
    starttime = time.time()
    myHealthCare(n)
    runtime = round((time.time() - starttime),2)
    return runtime

def benchmarkingAll(n):
    starttime = time.time()
    myHealthCare(n)
    abnormalSignAnalytics(abnMeasure, abnColumn)
    frequencyAnalytics(freqMeasure)
    healthAnalyser1(value)
    runtime = round((time.time() - starttime),2)
    return runtime


print(" For n=1,000; Time to run for myHealthCare:",round(benchmarkingmyHealthcare(1000),2), ". Time for all phases:",round(benchmarkingAll(1000),2))
print(" For n=2,500; Time to run for myHealthCare:",round(benchmarkingmyHealthcare(2500),2), ". Time for all phases:",round(benchmarkingAll(2500),2))
print(" For n=5,000; Time to run for myHealthCare:",round(benchmarkingmyHealthcare(5000),2), ". Time for all phases:",round(benchmarkingAll(5000),2))
print(" For n=7,500; Time to run for myHealthCare:",round(benchmarkingmyHealthcare(7500),2), ". Time for all phases:",round(benchmarkingAll(7500),2))
print(" For n=10,000; Time to run for myHealthCare:",round(benchmarkingmyHealthcare(10000),2), ". Time for all phases:",round(benchmarkingAll(10000),2))

print()
print("--------------------END--------------------------")


