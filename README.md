# pytorch_dct
## network中包含用到的网络：
cat后经过1*1卷积：resnet_conv_con1；  
特征直接相加：resnet_sum1.py；  
cat后经过conv、bn、relu：resnet_bn_con.py；  
多个block cat：resnet_con_12.py。  
## 训练流程：
利用creat_data.py生成img数据对应的频谱数据，利用creat_txt.py生成img和dct数据位置的txt文件，逐行读取txt文件，进行训练
