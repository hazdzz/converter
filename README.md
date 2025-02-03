# Converter

## About
The official PyTorch implementation of Converter proposed by our paper *Converter: Converting Transformers into DGNNs*.

## Citation
```
@article{zhang2025converter,
    author  = {Zhang, Jie and Wang, Kuan-Chieh and Chiu, Bo-Wei and Sun, Min-Te},
    title   = {Converter: Converting Transformers into DGNNs Form},
    journal = {arXiv preprint},
    year    = {2025}
}
```

## Datasets
1. LRA: https://mega.nz/file/tBdAyCwA#AvMIYJrkLset-Xb9ruA7fK04zZ_Jx2p7rdwrVVaTckE
2. GLUE: https://mega.nz/file/FZMygJiA#Z6rIO1kN_amRiCjdBbe2VDoDVwaW5jIA5WtABLil58Q

## Training Steps
1. Create a data folder:
```console
mkdir data
```

2. Download the dataset compressed archive
```console
wget $URL
```

3. Decompress the dataset compressed archive and put the contents into the data folder
```console
unzip $dataset.zip
mv $datast ./data/$datast
```

4. Run the main file
```console
python $dataset_main.py --task="$task"
```

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt

> [!Attention]
> In order to reproduce the experimental results, we strongly recommand you to adopt the following environment:
> python ~= 3.10.14
> cuda ~= 12.1