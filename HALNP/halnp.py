
from pyhanlp import *
import re


class NerByHanlp:
    def __init__(self, data_path, result_path):
        self.tags = ['product_name', 'company_name', 'person_name', 'location', 'org_name', 'time']
        self.data_path = data_path
        self.result_path = result_path
        self.d_pre = {'person_name':[], 'location':[], 'org_name':[]}

    def nerHanlp(self):
        NER = HanLP.newSegment().enablePlaceRecognize(True).enableNameRecognize(True).enableOrganizationRecognize(True)
        pattern = re.compile(r'(/[A-Za-z0-9]+)')

        rfile = open(self.data_path, encoding='UTF-8')

        data = rfile.readlines()

        for s in data:
            p = NER.seg(s)
            for i in p:
                word, tag = pattern.split(str(i))[:-1]
                self.result_write(word, tag)

        rfile.close()

    def result_write(self, word, tag):
        #wfile = open(self.result_path, 'w', encoding='UTF-8')
        
        if tag == '/nr' or tag == '/nr1' or tag == '/nrj' or tag == '/nr2' or tag == '/nrf':
            word = '{{person_name:' + word + '}}'
            self.d_pre['person_name'].append(word)
        elif tag == '/ns' or tag == '/nsf':
            word = '{{location:' + word + '}}'
            self.d_pre['location'].append(word)
        elif tag == '/nt' or tag == '/ni' or tag == '/nic' or tag == '/nis' or tag == '/nit':
            word = '{{org_name:' + word + '}}'
            self.d_pre['org_name'].append(word)

        #wfile.write(word)

        #wfile.close()

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
        RPF[key]['P']=RPF[key]['c']/RPF[key]['all_true']
        RPF[key]['F']=2*(RPF[key]['R']*RPF[key]['P'])/(RPF[key]['R']+RPF[key]['P'])
        print('%s:\nrecall:%f   precision:%f    F-measure:%f\n'%(key,RPF[key]['R'],RPF[key]['P'],RPF[key]['F']))



if __name__ == '__main__':
    nbr = NerByHanlp('data0.txt',
                     'result_hanlp.txt')

    nbr.nerHanlp()
    f = open('BosonNLP_NER_6C.txt','r',encoding='utf-8')
    data = f.read()
    f.close()
    d_true = {}
    #d_true['product_name'] = re.findall(r'{{product_name:[^\}]+}}',data)
    d_true['person_name'] = re.findall(r'{{person_name:[^\}]+}}',data)
    d_true['location'] = re.findall(r'{{location:[^\}]+}}',data)
    d_true['org_name'] = re.findall(r'{{org_name:[^\}]+}}',data)
    #d_true['time'] = re.findall(r'{{time:[^\}]+}}',data)
    #d_true['company_name'] = re.findall(r'{{company_name:[^\}]+}}',data)
    get_RPF(d_true,nbr.d_pre)