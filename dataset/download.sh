git clone https://www.modelscope.cn/datasets/tany0699/imagenet_val.git
wget -O ./imagenet_val.zip 'https://dataset-hub.oss-cn-hangzhou.aliyuncs.com/public-unzip-dataset/tany0699/imagenet_val/master/imagenet_val.zip?Expires=1736196594&OSSAccessKeyId=LTAI5tAoCEDFQFyV5h8unjt8&Signature=64xnC6qm%2BdfhypH%2Fr73xcku4unc%3D&response-content-disposition=attachment%3B'
unzip -j ./imagenet_val.zip -d ./data
rm ./imagenet_val.zip
python ./data_process.py