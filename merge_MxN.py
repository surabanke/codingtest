#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:04:37 2022

@author: eunhwalee
"""
# merge img MxN

import numpy as np
import cv2
import os

def vertical_match(add_img,root,origin_img):
    global can
    can = 0
    
    if (origin_img.shape[0])%(add_img.shape[0]) != 0:
        add_img = cv2.rotate(add_img,cv2.ROTATE_90_COUNTERCLOCKWISE) 
    
    if (root.shape[0]) % (add_img.shape[0]) != 0:
        root = cv2.rotate(root,cv2.ROTATE_90_COUNTERCLOCKWISE) 

       
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

            c_3 = cv2.vconcat([root_lst[i],add_lst[j]]) 
            c_4 = cv2.vconcat([add_lst[j],root_lst[i]])
            candidate = [c_3,c_4]
    
            for z in range(len(candidate)):
                res = cv2.matchTemplate(origin_img, candidate[z], cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)

                if maxv >= 0.98:
                    can = candidate[z]
                        
                    break

                        
    return can
            


def horizontal_match(root,add_img,origin_img):
    global can
    can = 0
    candidate = []
    c_1 = cv2.hconcat([root,add_img]) # merge images
    c_2 = cv2.hconcat([add_img,root])

    candidate = [c_1,c_2]
    
    for z in range(len(candidate)):
        res = cv2.matchTemplate(origin_img, candidate[z], cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv >= 0.98:
            can = candidate[z]
            break
        else:
            continue
    return can
        



column = 3
row = 3


fin_lst = []
final_img = []
totalM_N = [] 
    

    
path =  "/Users/eunhwalee/Desktop/divide_image/"
files = os.listdir(path)
    
origin_img = cv2.imread('/Users/eunhwalee/Desktop/img.png')
files.remove(files[1])
    
    
for i in range(column):

    root = cv2.imread(path+files[0])
    files.remove(files[0])
    print(i)
    for k in range(row - 1):

        iter_num = len(files)

        for j in range(iter_num):
                
            add_img = cv2.imread(path + files[j])
            res = vertical_match(add_img, root,origin_img)  # vconcat
            if (type(res) is np.ndarray) == True:
                files.remove(files[j])
                fin_lst.append(res)
                root = res
                break
        if origin_img.shape[0] == root.shape[0]:
            totalM_N.append(fin_lst[-1])
                        

root = totalM_N[0]        
del totalM_N[0]
for x in range(len(totalM_N)):

    iter_num = len(totalM_N)
        
    for y in range(iter_num):
        add_img = totalM_N[y]
        res = horizontal_match(add_img,root,origin_img)
            
        if (type(res) is np.ndarray) == True:
            del totalM_N[y]
            final_img.append(res)
            root = res
            break

final_img = cv2.imwrite(path + 'final_img.png',final_img[-1])

    
    
    
    
















