import re

rfile = open('BosonNLP_NER_6C.txt', encoding='UTF-8')
wfile = open('hmm_train_data.txt', 'w', encoding='UTF-8')

data = rfile.readlines()

tags = ['product_name', 'company_name', 'person_name', 'location', 'org_name', 'time']
labels = ['PRO', 'COM', 'PER', 'LOC', 'ORG', 'TIM']

pattern = re.compile(r'({{\w*_*\w*:[^}]*}})')

for s in data:
    sw = ''
    text_list = []
    label_list = []
    j = pattern.split(s)
    for word in j:
        if pattern.match(word) != None:
            tag = word[2:word.index(':')]
            w = word[word.index(':') + 1:word.index('}')]
            l = labels[tags.index(tag)]
            text_list.append(w[0])
            label_list.append('B-'+l)
            for i in w[1:]:
                text_list.append(i)
                label_list.append('I-'+l)
        else:
            for i in word:
                text_list.append(i)
                label_list.append('O')
    wfile.write('{\'text\': '+str(text_list)+',\'label\': '+str(label_list)+'}\n')

rfile.close()
wfile.close()
