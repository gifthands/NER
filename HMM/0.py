# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:41:15 2021

@author: Administrator
"""

import re

f = open("BosonNLP_NER_6C.txt","r",encoding='utf-8')   #设置文件对象
data = f.read()     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭
#所有类型的数据
product_name = re.findall(r'\{\{product_name:[\u4e00-\u9fa5|\w]+}}',data)
person_name = re.findall(r'\{\{person_name:[\u4e00-\u9fa5|\w]+}}',data)
location = re.findall(r'\{\{location:[\u4e00-\u9fa5|\w]+}}',data)
org_name = re.findall(r'\{\{org_name:[\u4e00-\u9fa5|\w]+}}',data)
time = re.findall(r'\{\{time:[\u4e00-\u9fa5|\w]+}}',data)
company_name = re.findall(r'\{\{company_name:[\u4e00-\u9fa5|\w]+}}',data)

d = {'1':[]}
d['1'].append(1)
d['1'].append(1)
print(d['1'])