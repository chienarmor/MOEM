# 打开原始文本文件
with open('/mnt/sda3/ztj/MoEM/data/unified/seed5/test.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
# 创建两个空字典来存储分割后的内容
laptop14_content = []
rest14_content = []
rest15_content = []
rest16_content = []
# 遍历每一行，根据关键字分割内容
for line in lines:
    if 'laptop14' in line:
        laptop14_content.append(line)
    elif 'rest14' in line:
        rest14_content.append(line)
    elif 'rest15' in line:
        rest15_content.append(line)
    elif 'rest16' in line:
        rest16_content.append(line)
# 将分割后的内容写入新的文本文件
with open('data_my/aste/unified/laptop14.txt', 'w', encoding='utf-8') as laptop_file:
    laptop_file.writelines(laptop14_content)
with open('data_my/aste/unified/rest14.txt', 'w', encoding='utf-8') as rest_file:
    rest_file.writelines(rest14_content)
with open('data_my/aste/unified/rest15.txt', 'w', encoding='utf-8') as rest_file:
    rest_file.writelines(rest15_content)
with open('data_my/aste/unified/rest16.txt', 'w', encoding='utf-8') as rest_file:
    rest_file.writelines(rest16_content)
print("文档已成功分割为laptop14.txt和rest14.txt。")