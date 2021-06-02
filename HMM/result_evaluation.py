import re


class Evaluation:
    def __init__(self, a_path, r_path):
        self.data_answer = open(a_path, encoding='UTF-8').readlines()
        self.data_result = open(r_path, encoding='UTF-8').readlines()

    def nerSplit(self, s):
        pattern = re.compile(r'({{\w*_*\w*:[^}]*}})')
        w = []
        t = []
        l = []
        sr = ''
        j = pattern.split(s)
        for word in j:
            if pattern.match(word) is not None:
                t.append(word[2:word.index(':')])
                w.append(word[word.index(':') + 1:word.index('}')])
                l.append(len(sr))
                sr += word[word.index(':') + 1:word.index('}')]
            else:
                sr += word

        return [w, t, l]

    def compare(self):
        answer_num = 0
        result_num = 0
        right_num = 0
        exact_num = 0
        rtag_num = []
        tags = ['product_name', 'company_name', 'person_name', 'location', 'org_name', 'time']
        for i, j in zip(self.data_answer, self.data_result):
            answer = self.nerSplit(i)
            result = self.nerSplit(j)
            answer_num += len(answer[0])
            result_num += len(result[0])

            # 先确认位置，如果类别正确、长度不足，加正确率，不加准确率
            for rn in range(len(result[0])):
                if result[2][rn] in answer[2]:
                    n = answer[2].index(result[2][rn])
                    if result[1][rn] == answer[1][n]:
                        if result[0][rn] == answer[0][n]:
                            right_num += 1
                            exact_num += 1
                        elif result[0][rn] in answer[0][n]:
                            right_num += 1

        result_list1 = [right_num / answer_num, exact_num / answer_num, right_num / result_num, exact_num / result_num]
        result_list2 = []

        return result_list1


if __name__ == '__main__':
    na = Evaluation('BosonNLP_NER_6C.txt',
                    'result_hanlp.txt')
    print(na.compare())
