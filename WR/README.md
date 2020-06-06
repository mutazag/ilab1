# files 

all files are in drop box here: [dropbox for WR](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAAE2yMC7n3tcf7M8IhirU1ja?dl=0&lst=)

## download files

the following csvs are in the root of the drop box folder: 
- [.csv](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AADWWMmVCBD1dOUaT7Xb2jBja/.csv?dl=0)
- [1l_atomicPLMF_6138structures.csv](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAA-hmBN3DVYFgrZ4ahPlUpSa/1l_atomicPLMF_6138structures.csv?dl=0)
- [COMPLETE_DL_IE_PREDICTED.sv.csv](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AADOWYyXiYC7ZQrI-P7tZxj7a/COMPLETE_DL_IE_PREDICTED.sv.csv?dl=0)
- [C33_DFT.csv](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAA8iMufz6NWuEIaAT_iyh4xa/C33_DFT.csv?dl=0)
- [IE_DFT.csv](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAA0gz2xUDlCJl_chaOanBXya/IE_DFT.csv?dl=0)
- [PLMF.csv](https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAAFqeBteVKLwM4yYnzDGPH6a/PLMF.csv?dl=0)


downloading the files to folder `data/WR`

```
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AADWWMmVCBD1dOUaT7Xb2jBja/.csv?dl=0" --output nameless.csv
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAA-hmBN3DVYFgrZ4ahPlUpSa/1l_atomicPLMF_6138structures.csv?dl=0" --output 1l_atomicPLMF_6138structures.csv
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AADOWYyXiYC7ZQrI-P7tZxj7a/COMPLETE_DL_IE_PREDICTED.sv.csv?dl=0" --output COMPLETE_DL_IE_PREDICTED.sv.csv
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAA8iMufz6NWuEIaAT_iyh4xa/C33_DFT.csv?dl=0" --output C33_DFT.csv
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAA0gz2xUDlCJl_chaOanBXya/IE_DFT.csv?dl=0" --output IE_DFT.csv
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AAAFqeBteVKLwM4yYnzDGPH6a/PLMF.csv?dl=0" --output PLMF.csv

```



# download files 

```
curl -L "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AACQd3nAgPV3ASVq4EFlrhwra/LASSO_BR2_1?dl=0&preview=1l_atomicPLMF_6138structures.csv" --output "./data/LASSO_BR2_1/1l_atomicPLMF_6138structures.csv"

```

file split for easier management 

```
cd data/LASSO_BR2_1
mkdir split

split -l 1000000 -d --additional-suffix=.csv --suffix-length=5  1l_atomicPLMF_6138structures.csv ./split/f_


wc -l split/*
```