# Converter

## About
The official PyTorch implementation of Converter proposed by our paper *Converter: Converting Transformers into DGNNs*.

## Citation
```
@inproceedings{zhang2024converter,
    author  = {Zhang, Jie and Wang, Kuan-Chieh and Chiu, Bo-Wei and Sun, Min-Te},
    title   = {Converter: Converting Transformers into DGNNs},
    journal = {arXiv preprint},
    year    = {2024}
}
```

## Datasets
1. LRA: https://mega.nz/file/sdcU3RKR#Skl5HomJJldPBqI7vfLlSAX8VA0XKWiQSPX1E09dwbk

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt

## Training Steps
1. Create a data folder:
```console
mkdir data

2. Download the dataset compressed archive
```console
wget $URL

3. Decompress the dataset compressed archive and put the contents into the data folder
```console
unzip $dataset.zip
mv $datast ./data/$datast

4. Run the main file
```console
python $dataset_main.py