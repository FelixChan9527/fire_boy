import numpy as np
import os
import random
import time


path = 'E:/photo/'
# 生成随机数整数，用于随机删除文件，参数一为下限、参数二位上限、参数三为随机数list大小
# random_file_num = np.random.randint(0, 3320, 2000)
random_file_num = random.sample(range(0, 1881), 1000)
print(len(random_file_num))


for i in random_file_num:
    try:
        file_name = os.listdir(path)[i]
    except IndexError:
        print("error in ", file_name)
        continue
        pass
    # print(file_name)
    path_file_name = os.path.join(path + file_name)
    exit_file = os.path.exists(path_file_name)
    if exit_file is True:
        os.remove(path_file_name)
        print("delete the ", path_file_name)
        time.sleep(0.1)
        # continue
        # pass

# print(len(path_name))


