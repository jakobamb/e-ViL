# requirements: 
#   - have Kaggle credentials available, 
#     see https://github.com/Kaggle/kaggle-api#api-credentials
#   - python3, pip3

echo "Installing GDown via PIP"
pip install gdown
echo "Installing Kaggle CLI"
pip install kaggle
echo "Downloading: Flickr30k from Kaggle"
kaggle datasets download hsankesara/flickr-image-dataset

echo "Downloading: COCOcaption package for automatic NLG metrics"
# placed in root
gdown --fuzzy https://drive.google.com/file/d/1nLlrtQlsP5kSeB9L0PTle4PeGL9CJg29/view
unzip cococaption.zip
rm cococaption.zip

echo "Downloading: [e-SNLI-VE] Faster-RCNN features for Flickr30k"
mkdir -p download
./download-tools/download_ve.sh download
mv download/img_db data/esnlive/img_db

echo "Downloading: [e-SNLI-VE] Faster-RCNN features for Flickr30k - json files"
gdown --fuzzy https://drive.google.com/file/d/1kGGunXYCtX7sYgjhbIMrtI8GcLa7Bg97/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/16PQ9AHdSRKT65Bk8zhdtQdPhnTQejjED/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1yKCUANkKZswdGX4xtESMhgzK0wWPd89M/view?usp=sharing

mkdir data/esnlive/
mv esnlive_dev.json data/esnlive/
mv esnlive_test.json data/esnlive/
mv esnlive_train.json data/esnlive/

echo "Downloading: [VQA-X] Faster-RCNN features for MS COCO"
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/fasterRCNN_features
unzip data/fasterRCNN_features/train2014_obj36.zip -d data/fasterRCNN_features && rm data/fasterRCNN_features/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/fasterRCNN_features
unzip data/fasterRCNN_features/val2014_obj36.zip -d data && rm data/fasterRCNN_features/val2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data/fasterRCNN_features
unzip data/fasterRCNN_features/test2015_obj36.zip -d data && rm data/fasterRCNN_features/test2015_obj36.zip

echo "Downloading: [VQA-X] VQA-X dataset"
gdown --fuzzy https://drive.google.com/file/d/1P3hSie3osvy2_tcjgMrgy4ZulkkXktbt/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1b3x4ku3LlOGEoFPQiVjDscIy4Spn-UGJ/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1PgGqu8R9tzWReJW4epsLZ5VNXIE_dxW_/view?usp=sharing
mkdir data/vqax
mv test_x.json data/vqax/
mv train_x.json data/vqax/
mv val_x.json data/vqax/

echo "Downloading: [VCR] Faster-RCNN features"
./download-tools/download_vcr.sh download
mv download/img_db data/vcr

echo "Downloading: [UNITER] genral pretrained base"
wget https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-base.pt
wget https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-base-vcr_2nd_stage.pt