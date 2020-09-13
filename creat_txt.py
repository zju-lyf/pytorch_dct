import os
import os.path

def get_files_list(dir1,dir2):
    images_list=[]
    for parent,dirnames,imagesnames in os.walk(dir1):
        for imagename in imagesnames:
            curr_file=parent.split(os.sep)[-1]
            print(curr_file)
            if curr_file=='B_fake':
                labels=0
            elif curr_file=='A_real':
                labels=1
            cur_image_path=dir1+'/'+curr_file+'/'+imagename
            cur_dct_path=dir2 + '/'+curr_file + '/' +imagename
            images_list.append([cur_image_path,cur_dct_path,labels])
    return images_list

def write_txt(content,filename,mode='w'):
    with open(filename,mode) as f:
        for line in content:
            str_line=""
            for col,data in enumerate(line):
                if not col==len(line)-1:
                    str_line=str_line+str(data)+" "
                else:
                    str_line=str_line+str(data)+"\n"
            f.write(str_line)



if __name__=='__main__':
    train_dir1='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img/train'
    train_dir2='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/dct/train'
    train_txt='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img_dct/train.txt'
    train_data=get_files_list(train_dir1,train_dir2)
    write_txt(train_data,train_txt,mode='w')

    validation_dir1='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img/val'
    validation_dir2='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/dct/val'
    validation_txt='/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img_dct/val.txt'
    validation_data=get_files_list(validation_dir1,validation_dir2)
    write_txt(validation_data,validation_txt,mode='w')

    test_dir1 = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img/test'
    test_dir2 = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/dct/test'
    test_txt = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img_dct/test.txt'
    test_data = get_files_list(test_dir1, test_dir2)
    write_txt(test_data, test_txt, mode='w')
