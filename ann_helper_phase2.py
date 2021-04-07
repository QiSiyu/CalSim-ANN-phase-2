#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:59:10 2019

@author: siyuqi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

def normalize_in(ori_data,hi=0.9,lo=0.1,masked_entry_value=None):
    if masked_entry_value is not None:
        mask = ori_data!=masked_entry_value
    else:
        mask = np.ones(ori_data.shape, dtype=bool)

    data_max = np.max(ori_data[mask],axis=0)
    data_min = np.min(ori_data[mask],axis=0)

    norm_slope = np.divide((hi-lo), 
                           (data_max-data_min), 
                           out=np.zeros_like(data_max),
                           where=(data_max-data_min)!=0)
    norm_bias = lo - norm_slope * data_min
    return [(norm_slope * ori_data + norm_bias)*mask,norm_slope,norm_bias]


def Filter(string, substr): 
    return [ii for ii in range(len(string))
        if not any(sub.lower() in string[ii].lower() for sub in substr)]        

   
def read_csv_with_mask(input_data_path,output_data_path,input_var_list,output_var_list, masked_station_month_pair={}):
    inputs = pd.read_csv(input_data_path,header=0,index_col=0,comment='#')
    # select input parameters
    drop_idx = Filter(list(inputs.columns.values),input_var_list)
    inputs.drop(columns=drop_idx,inplace=True)
    # sort input data by input_var_list
    inputs.reindex(columns=input_var_list)
    inputs=inputs.fillna(0)
    
    outputs = pd.read_csv(output_data_path,header=0,index_col=0,comment='#')
    outputs.index = pd.to_datetime(outputs.index)
    

    # select output station
    drop_idx = Filter(list(outputs.columns.values),output_var_list)
    outputs.drop(columns=outputs.columns[drop_idx],inplace=True)

    # sort outputs data by output_var_list
    outputs.reindex(columns=output_var_list)

    # mask out output entries
    for station in masked_station_month_pair.keys():
        if station not in output_var_list:
            continue
        for masked_year, masked_month in masked_station_month_pair[station]:
            outputs.loc[(outputs.index.year==masked_year) & (outputs.index.month==masked_month), station] = 0
    
            
    return [np.array(inputs).astype('float32'),
            np.array(outputs).astype('float32')]

def process_data_vary_pred(input_dataset,output_dataset,
                           single_days=7, window_num=10,
                           window_size = 11,predict_list=None):
    assert len(input_dataset)==len(output_dataset), "input and output data must have same length"
    assert len(predict_list)==output_dataset.shape[1]
    assert min(predict_list) >= 0
    data = []
    labels = []
    total_avg_num = window_num*window_size
    end_date = len(input_dataset)-max(predict_list)
    if window_num != 0 and window_size != 0:
        for i in range(single_days+total_avg_num-1, end_date):
            # Reshape data from (history_size,# of variables) to
            # new_shape
            data.append(np.concatenate((np.reshape(input_dataset[i:i-single_days:-1], (single_days,-1)),
                                        np.mean(np.reshape(input_dataset[i-single_days:i-single_days-total_avg_num:-1], (window_num,window_size,-1)),axis=1)),axis=0))
            labels.append(np.asarray([output_dataset[i+predict,ii] for ii,predict in enumerate(predict_list)]))
    else:
        for i in range(single_days+total_avg_num-1, end_date):
            # Reshape data from (history_size,# of variables) to
            # new_shape
            if i-single_days==-1:
                data.append(np.reshape(input_dataset[i::-1], (single_days,-1)))
            else:
                data.append(np.reshape(input_dataset[i:i-single_days:-1], (single_days,-1)))
            labels.append(np.asarray([output_dataset[i+predict,ii] for ii,predict in enumerate(predict_list)]))
            
    return np.array(data).transpose(0,2,1), np.array(labels)

  
def conv_filter_generator(single_days=7,window_num=10,window_size = 11):
    w = np.zeros((1,single_days+window_num*window_size,single_days+window_num))
    for ii in range(single_days):
        w[0,single_days+window_num*window_size-ii-1,single_days-ii-1] = 1
    for ii in range(window_num):
        w[0,((window_num-ii-1)*window_size):((window_num-ii)*window_size),single_days+ii] = 1/window_size
    return w
  
def conv_filter_generator_v2(single_days=7,window_num=10,window_size = 11,var_num=7):
    w1 = np.repeat(np.identity(var_num)[np.newaxis,...],1, axis=0)
    w2 = np.repeat(np.identity(var_num)[np.newaxis,...]/window_size,window_size, axis=0)
    return (w1,w2)
  
def add_noise(data, noise_dict):
    ## add noise to data according to noise_dict
    ## data must have the shape of (no. of samples, no. of variables)
    noisy_data = np.copy(data)
    p_signal = np.mean(data**2,axis=0)
    for index in noise_dict.keys():
        p_noise = p_signal[index] / (10**(noise_dict[index]/10))
        noise = np.random.normal(0, np.sqrt(p_noise), data.shape[0])
        noisy_data[:,index] = noise+data[:,index]
    return noisy_data