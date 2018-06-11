import sys
import glob
import numpy as np
from keras.preprocessing import image
import pandas
from sklearn import preprocessing
from keras.backend import to_categorical
threshold=10

def read_filenames(image_path):
    filenames=[]
    image_path=image_path+"/*.jpg"
    for filename in glob.glob(image_path):
        if filename not in cfiles:
            filenames.append(int(filename[39:-4]))
    images=[]
    for i in filenames:
        images.append(filenames[len(sys.argv[1]+1):])
    return images,filenames


def read_data(info_path):
    data=pd.read_csv(info_path)
    le=preprocessing.LabelEncoder()
    data['artist']=le.fit_transform(data['artist'])
    return data

def find_artist_from_data(data,images):
    artist=[]
    i=0
    for image in images:
            i+=1
            index=data["filename"][data["filename"]==image].index[0]
            artist.append(data.at[index,"artist"])
            s="finding artist of image "+str(i)
            print(s, end='\r')
    return artist

def remove_images_less_than_threshold(artist,images,filenames,threshold):
    le=preprocessing.LabelEncoder()
    artist_le=le.fit_transform(artist)
    no_count={}
    index=0
    for i in artist_le:
        count=0
        index+=1
        for j in artist_le:
            if i==j:
                count+=1
        no_count[str(i)]=count
    count=0
    ind=[]
    use=0
    for k,v in no_count.items():
        count+=1
        artist_index=[]
        if v<=threshold:
            for i in range(len(artist_le)):
                if artist_le[i]==int(k):
                    artist_index.append(i)
            for t in artist_index:
                ind.append(t)
        else:
            use+=1
    for index in sorted(ind,reverse=True):
        del filenames[index]
        del images[index]
        s=str(index)+" is done"
        print(s, end='\r')
    return images,filenames

def pixel_converter(start,end,filenames):
    pixels=[]
    i=start
    while(i<=end):
        img = image.load_img(filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x=x/255
        pixels.append(x)
        s="convertng image "+str(i)+" to pixels"
        print(s, end='\r')
        i+=1
    pixels=np.array(pixels)
    return pixels

def to_categorical(artist):
    le=preprocessing.LabelEncoder()
    artist_le=le.fit_transform(artist)
    one_hot_artist=to_categorical(artist_le)
    one_hot_artist=subset_artist.reshape(len(artist),1,1,max(artist_le))
    return one_hot_artist

def main(image_path,info_path):
    images,filenames=read_filenames(image_path)
    data=read_data(info_path)
    artist=find_artist_from_data(data,images)
    images,filenames=remove_images_less_than_threshold(artist,images,filenames,threshold)
    pixels=pixel_converter(0,len(filenames)-1,filenames)
    one_hot_artist=to_categorical(artist)
    np.save('X',pixels)
    np.save('y',one_hot_artist)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 preprocess.py `image_path info_path')
    else:
        main(sys.argv[1], sys.argv[2])
