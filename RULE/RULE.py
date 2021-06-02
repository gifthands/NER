# -*-coding:utf-8-*-
import re

def get_RPF(d_true, d_pre):
    #print(d_true)
    #print(d_pre)
    RPF = {}
    for key in d_true:
        RPF[key] = {'c':0, 'all_true':0,'all_pre':0,'R':0,'P':0,'F':0}
    a = d_true
    b = d_pre
    '''
    correct = 0
    incorrect = 0
    all_pre = 0     #预测出的所有词的个数
    all_true = 0    #实际所有词的个数
    '''
    for key in a:
        for x in a[key]:
            RPF[key]['all_true'] += 1
    for key in b:
        for x in b[key]:
            RPF[key]['all_pre'] += 1
            if x in a[key]:
                RPF[key]['c'] += 1
                a[key].remove(x)   #去除已经判断出的
    
    for key in RPF:
        RPF[key]['R']=RPF[key]['c']/RPF[key]['all_pre']
        RPF[key]['P']=2*RPF[key]['c']/RPF[key]['all_true']
        RPF[key]['F']=2*(RPF[key]['R']*RPF[key]['P'])/(RPF[key]['R']+RPF[key]['P'])
        print('%s:\nrecall:%f   precision:%f    F-measure:%f\n'%(key,RPF[key]['R'],RPF[key]['P'],RPF[key]['F']))

f = open('BosonNLP_NER_6C.txt','r',encoding='utf-8')
data = f.read()
f.close()
d_true = {}
#d_true['product_name'] = re.findall(r'{{product_name:[^\}]+}}',data)
#d_true['person_name'] = re.findall(r'{{person_name:[^\}]+}}',data)
d_true['location'] = re.findall(r'{{location:[^\}]+}}',data)
d_true['org_name'] = re.findall(r'{{org_name:[^\}]+}}',data)
d_true['time'] = re.findall(r'{{time:[^\}]+}}',data)
#d_true['company_name'] = re.findall(r'{{company_name:[^\}]+}}',data)

true_all = len(d_true['location'])+len(d_true['org_name'])

f = open('data0.txt','r',encoding='utf-8')
data0 = f.read()
f.close()

f1 = open('location.txt','r',encoding='utf-8')
loc = f1.read().splitlines()
f1.close()

f2 = open('organize.txt','r',encoding='utf-8')
org = f2.read().splitlines()
f2.close()

d_pre = {}
d_pre['location'] = []
d_pre['org_name'] = []

pattern = r'(春|(春天)|夏|(夏天)|秋|(秋天)|冬|(冬天)|([上下(上半)(下半)今昨明后早每]+(年|月|天|日)))|(([0-9零一二两三四五六七八九十]+年) \
    ?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([0-9零一二两三四五六七八九十]+[点时])? \
        ([0-9零一二三四五六七八九十百]+[分钟])?([0-9零一二三四五六七八九十百]+秒)?)|(第[一二三四]季度)|(周[一二三四五六七])'
time_list = re.findall(pattern, data0)

time=[]
for i in time_list:
    for j in i:
        if(j!=''):
            time.append('{{time:%s}}'%j)
d_pre['time'] = time


f = open('time_pre.txt','w')
f.write(str(time))

#print(time)
#d_pre['time'] = ['{{time:%s}}'%x for x in time_list]
for x in loc:
    if(x in data0):
        d_pre['location'].append('{{location:%s}}'%x)
for y in org:
    if( y in data0):
        d_pre['org_name'].append('{{org_name:%s}}'%y)

get_RPF(d_true,d_pre)
