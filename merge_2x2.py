#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:27:37 2022

@author: eunhwalee
"""

# merge img 2x2

import numpy as np
import cv2
import os

global can

img = cv2.imread("/Users/eunhwalee/Desktop/sibadog.png")




def matching_and_merge(add_img,root,origin_img):
    
    if origin_img.shape[0]%(root.shape[0]) != 0:
        root = cv2.rotate(root,cv2.ROTATE_90_COUNTERCLOCKWISE) 
        
    if origin_img.shape[0]%(add_img.shape[0]) != 0:
        add_img = cv2.rotate(add_img,cv2.ROTATE_90_COUNTERCLOCKWISE) 
       
    root_f = cv2.flip(root,0)
    root_fm = cv2.flip(root_f,1)
    root_m = cv2.flip(root,1)
    root_mf = cv2.flip(root_m,0)
    
    add_f = cv2.flip(add_img,0)
    add_m = cv2.flip(add_img,1)
    
    add_mf = cv2.flip(add_m,0)
    add_fm = cv2.flip(add_f,1)
        
        
    root_lst = [root,root_m,root_f,root_mf,root_fm]
    add_lst = [add_img,add_f,add_m,add_mf,add_fm]
        
    for i in range(len(root_lst)):
        
        for j in range(len(add_lst)):

            c_1 = cv2.hconcat([root_lst[i],add_lst[j]]) # merge images
            c_2 = cv2.hconcat([add_lst[j],root_lst[i]])
            c_3 = cv2.vconcat([root_lst[i],add_lst[j]]) 
            c_4 = cv2.vconcat([add_lst[j],root_lst[i]])
            candidate = [c_1,c_2,c_3,c_4]
    
            for z in range(len(candidate)):
                res = cv2.matchTemplate(origin_img, candidate[z], cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)

                if maxv >= 0.98:
                    can = candidate[z]
                        
                    break
                else:
                    continue
                        
    return can
            




def merge_img(img,column,row,outputfilename):  # main function

    fin_lst = []


    
    path =  "/Users/eunhwalee/Desktop/divide_image/"
    files = os.listdir(path)
    root = cv2.imread(path+files[0])
    files.remove(files[0])
    
    origin_img = cv2.imread('/Users/eunhwalee/Desktop/img.png')
    
    for i in range(len(files)):
        add_img = cv2.imread(path + files[i])
        cann = matching_and_merge(add_img,root,origin_img)


        if (type(cann) is np.ndarray) == True:
            fin_lst.append(cann)
            files.remove(files[i])
            #can = 0
            break
        else:
            continue


    root = cv2.imread(path +files[0])
    #files.remove(files[0])
    add_img = cv2.imread(path + files[1])
    cann_1 = matching_and_merge(add_img,root,origin_img)
    fin_lst.append(cann_1)
    
    #final_img = matching_and_merge(fin_lst[0], fin_lst[1],origin_img)
    if 2*fin_lst[0].shape[0] == origin_img.shape[0]:
        candidate_1 = cv2.vconcat([fin_lst[0],fin_lst[1]])
        candidate_2 = cv2.vconcat([fin_lst[1],fin_lst[0]])
    else:
        candidate_1 = cv2.hconcat([fin_lst[0],fin_lst[1]])
        candidate_2 = cv2.hconcat([fin_lst[1],fin_lst[0]])
        
    candidate = [candidate_1,candidate_2]
    
    for z in range(len(candidate)):
        res = cv2.matchTemplate(origin_img, candidate[z], cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        #threshold = 0.89
        
        if maxv >= 0.98:
            final_img = cv2.imwrite(path + 'finat_img.png',candidate[z])
    
    
    return final_img
        
    
    
        
    

        
        
        
        
        
        
        