import os
from tqdm import tqdm

def dataset_list(dataset_path,dataset_list):
     label_list = os.listdir(dataset_path)
     f=open(dataset_list,'w')
     k=0

     for i in tqdm(label_list):
         label_path=os.path.join(dataset_path,i)
         if os.listdir(label_path):
            image_list=os.listdir(label_path)
            for j in image_list:
                image_path=os.path.join(label_path, j)
                f.write(image_path+'  '+str(k)+'\n')
         k=k+1
     f.close()

if __name__=='__main__':
    dataset = '/home/sung/dataset/Face/faces_webface_112x112/image'
    list = '/home/sung/dataset/Face/faces_webface_112x112/train.list'
    dataset_list(dataset, list)