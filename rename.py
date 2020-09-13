# -*- coding: utf-8 -*-
import os
import shutil
#设定文件路径
path='/home/zfy/Downloads/100/fake/'
fileList=os.listdir(path)
n = 0
print(fileList)
for i in fileList:
    foldpath = path + fileList[n]
    #print(foldpath)
    piclist = os.listdir(foldpath)
    #print(piclist)
    m = 0
    for j in piclist:
	# 设置旧文件名（就是路径+文件名）
        #oldname = path + os.sep + piclist[m]  # os.sep添加系统分隔符
        oldname = foldpath + os.sep +piclist[m]
        # 设置新文件名
        newname = foldpath + os.sep + '100_'+ str(fileList[n]) + '_' + str(m + 1) + '.png'
        newpath = '/home/zfy/env/py3_pytorch/ws/MesoNet-Pytorch/deepfake_database/test2/df' +os.sep  + '100_'+ str(fileList[n]) + '_' + str(m + 1) + '.png'
        m += 1
        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改
        shutil.copyfile(newname,newpath)
    #print(oldname)
    print(newname)
    print(newpath)
    n += 1


