# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:41:15 2021

@author: Administrator
"""

import re

f = open("BosonNLP_NER_6C.txt","r",encoding='utf-8')   #设置文件对象
data = f.read()     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭

data0 = re.sub(r'{{[^:]+:([^}]+)}}', lambda x: x.group(1), data)
f = open('data0.txt','w',encoding='utf-8')
f.write(data0)
