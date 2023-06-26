import matplotlib.pyplot as plt

file = open('./model/TU_Synapse256/TU_pretrain_R50-ViT-B_16_skip3_epo50_bs1_256/log.txt')  # 打开文档
data = file.readlines()  # 读取文档数据
# print(len(data))
# print(data)

# para_1 = []  # 新建列表，用于保存第一列数据
para_2 = []  # 新建列表，用于保存第二列数据

for num in data:
    # print(num)
    try:
        temp = num.split(':')
        # t = temp[4].split(',')
        x = temp[5].split('\n')
        # print(temp)
        # print(t[0])
        print(x)
    except:
        continue
    # para_1.append(float(num.split(':')[1].split(' ')[0]))
    # para_2.append(float(num.split(':')[2]))

    # para_1.append(float(t[0]))
    para_2.append(float(x[0]))
# print(para_1)

plt.figure()
plt.title('loss')
# plt.title('loss_ce')
# plt.plot(para_1)
plt.plot(para_2)
plt.show()