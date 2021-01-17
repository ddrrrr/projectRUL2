import numpy as np 
import pandas as pd 
from dataset import DataSet
import os

def Data2csv(name,export_path):
    dataset = DataSet.load_dataset(name)
    data = dataset.get_value("data")
    bearings_name = dataset.get_value("bearing_name")

    if (not os.path.exists(export_path)):
        os.makedirs(export_path)
    if (not os.path.exists(export_path+'/'+name)):
        os.makedirs(export_path+'/'+name)
    for i in range(len(data)):
        onedata = data[i]
        onename = bearings_name[i]
        if (not os.path.exists(export_path+'/'+name + '/' +onename)):
            os.makedirs(export_path+'/'+name+ '/' +onename)
        for j in range(onedata.shape[0]):
            onesignal = onedata[j,:,:]
            onesignal = np.int32(np.round(onesignal / 0.001)+2**15)
            np.savetxt(export_path+'/'+name+ '/' +onename + '/' + "%05d.csv" % j,
                        onesignal,delimiter=',',fmt='%d')
    return

if __name__ == '__main__':
    Data2csv("phm_data","./export_data")