import pandas as pd
import os
pwd = os.path.dirname(__file__)

def get_class_dict():
    with open(os.path.join(pwd, 'imagenet_val/classname.txt'), 'r') as f:
        classes = f.readlines()
    classes_dict = [c.split(',')[0].strip() for c in classes]
    classes_dict = dict(zip([i for i in range(1000)], classes_dict))
    return classes_dict

def process_csv():
    classes_dict = get_class_dict()
    df = pd.read_csv(os.path.join(pwd, 'imagenet_val/val.csv'))
    df['category'] = df['category'].astype(int)
    df['caption'] = df['category'].apply(lambda x: 'the ' + classes_dict[x])
    data_dir = os.path.join(pwd, 'data')
    df['link'] = df['image:FILE'].apply(lambda x: os.path.join(data_dir, os.path.basename(x)))
    df = df[df['link'].apply(lambda x: os.path.exists(x))]
    df = df[['link', 'caption']]

    # split train and val dataset
    df_val = df.groupby('caption').apply(lambda x: x.sample(n=5))
    df_val = df_val.reset_index(drop=True)
    df_val['type'] = df_val.index
    df_val['type'] = df_val['type'].apply(lambda x: x % 5)
    df_val = df_val.sort_values(by=['type'])                # comfirm the each batch read contains many category as possible 
    df_train = df[~df['link'].isin(df_val['link'])]
    return df_val, df_train

if __name__ == '__main__':
    csv_dir = os.path.join(pwd, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    df_val, df_train = process_csv()
    df_val.to_csv(os.path.join(csv_dir, 'val.csv'), index=False)
    df_train.to_csv(os.path.join(csv_dir, 'train.csv'), index=False)




