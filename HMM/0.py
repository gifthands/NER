# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:41:15 2021

@author: Administrator
"""

import re

f = open("BosonNLP_NER_6C.txt","r",encoding='utf-8')   #�����ļ�����
data = f.read()     #��txt�ļ����������ݶ��뵽�ַ���str��
f.close()   #���ļ��ر�
#�������͵�����
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