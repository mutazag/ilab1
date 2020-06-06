# ilab1
ilab1 project repository 

download data files from [link](https://www.dropbox.com/s/8kqnbfsdeven3x3/ML_IE_C33.zip?dl=0)

run the following commands from the data folder
```
  curl -L "https://www.dropbox.com/s/8kqnbfsdeven3x3/ML_IE_C33.zip?dl=0" --output ML_IE_C33.zip

  unzip ML_IE_C33.zip -d .
```



## missing values

[find missing values](https://medium.com/dunder-data/finding-the-percentage-of-missing-values-in-a-pandas-dataframe-a04fa00f84ab)

## counting number of records in data files 

``` 
$ wc -l *
  18,834,454 18M_PREDICTED.csv
  18834454 18M_small.csv
         0 300k_PREDICTED.csv
    296,836 300K_small.csv
      6583 IE_C33.png
      3655 IEC33_18M.png
     27792 ML-lub.pdf
  38003774 total
```

## split file

use `split` command to split by number of lines or file sizes

```
mkdir split
split -l 100 -d --additional-suffix=.csv --suffix-length=5  lasso_monolayer_data_C33.csv ./split/lasso_monolayer_data_C33_
wc -l split/*
```

note that first file will have a header, all other files will not include a header



## Plotting Heatmaps 

- Sample code matplotlib scatter plot [link](https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec)

- using plot.ly [link](https://plot.ly/python/heatmaps/)
- contour plots [link](https://plot.ly/python/contour-plots/)
- variety of 2D density plots [link](https://python-graph-gallery.com/2d-density-plot/)

### UMAP 

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data

1. The data is uniformly distributed on a Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected.

From these assumptions it is possible to model the manifold with a fuzzy topological structure. The embedding is found by searching for a low dimensional projection of the data that has the closest possible equivalent fuzzy topological structure.

#### Install UMAP
'python3.6 -m pip install numba'
'python3.6 -m pip install umap-learn --user'
reference [umap site](https://github.com/lmcinnes/umap)

```
@article{2018arXivUMAP,
     author = {{McInnes}, L. and {Healy}, J. and {Melville}, J.},
     title = "{UMAP: Uniform Manifold Approximation
     and Projection for Dimension Reduction}",
     journal = {ArXiv e-prints},
     archivePrefix = "arXiv",
     eprint = {1802.03426},
     primaryClass = "stat.ML",
     keywords = {Statistics - Machine Learning,
                 Computer Science - Computational Geometry,
                 Computer Science - Learning},
     year = 2018,
     month = feb,
}
```
## Saving pandas dataframe into pdf format

- using PDFKit 
- using Jinja and WeasyPrint [link](https://pbpython.com/pdf-reports.html)


### using PDF Kit 

``` {python}
import pandas as pd
import pdfkit as pdf
df.to_html('test.html')
PdfFilename='pdfPrintOut.pdf'
pdf.from_file('test.html', PdfFilename)
```

### getting weasyprint to work 

pip installing weasyprint alone does not install depdencies, especially cairo lib. 

refer to [WeasyPrint dependencies](https://weasyprint.readthedocs.io/en/stable/install.html#windows)

need to install [GTK 64bit](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer)

validate WeasyPrint setup by running `python -m weasyprint http://weasyprint.org weasyprint.pdf`


## multipprocessing parallal and  progress bars
 ### Progress Bar
 - several libraries [link](https://codingdose.info/2019/06/15/how-to-use-a-progress-bar-in-python/)
 - `tqdm` library does a good job to render progress bars 
 
 ### Parallel processing
 - `from joblib import Parallel', or 
 - `import multiprocessing`
 
 


## Useful stuff 

-[python 3 feaures and f string](https://datawhatnow.com/things-you-are-probably-not-using-in-python-3-but-should/) 

-[Download files in code using `requests`, `urllib` and `wget` libraries, and other examples](https://likegeeks.com/downloading-files-using-python/)

-[string format specs](https://www.dummies.com/programming/python/how-to-format-strings-in-python/)

## resizing linux VM disk 

first need to allocate additional disk storage in azure portal while VM is deallocated. 

When machine is started againwith the new disk size, need to repartition 

[azure guide for resizing linux os disk](https://blogs.msdn.microsoft.com/linuxonazure/2017/04/03/how-to-resize-linux-osdisk-partition-on-azure/)

[azure guide for resizing data disk](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/expand-disks)

other useful commands 


Resize the filesystem with resize2fs
```
mutaz@dsvmlinux:~$ resize2fs /dev/sdc1
resize2fs 1.42.13 (17-May-2015)
open: Permission denied while opening /dev/sdc1
mutaz@dsvmlinux:~$ sudo resize2fs /dev/sdc1
resize2fs 1.42.13 (17-May-2015)
Filesystem at /dev/sdc1 is mounted on /data; on-line resizing required
old_desc_blocks = 7, new_desc_blocks = 16
The filesystem on /dev/sdc1 is now 65429431 (4k) blocks long.

```

check partition sizes 
```
mutaz@dsvmlinux:~$ df -h
Filesystem      Size  Used Avail Use% Mounted on
udev             28G     0   28G   0% /dev
tmpfs           5.6G  9.0M  5.5G   1% /run
/dev/sda1        49G   26G   23G  54% /
tmpfs            28G     0   28G   0% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
tmpfs            28G     0   28G   0% /sys/fs/cgroup
/dev/sdc1       246G   88G  148G  38% /data
/dev/sdb1       335G   67M  318G   1% /mnt
tmpfs           5.6G     0  5.6G   0% /run/user/1003
```

devices and partitions 
```
mutaz@dsvmlinux:~$ lsblk
NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
sdb      8:16   0   340G  0 disk
+-sdb1   8:17   0   340G  0 part /mnt
sdc      8:32   0   250G  0 disk
+-sdc1   8:33   0 249.6G  0 part /data
sda      8:0    0    50G  0 disk
+-sda1   8:1    0    50G  0 part /

```


## Run jupyter in the background 

[Running Jupyter Notebook in the background](https://medium.com/@jim901127/running-jupyter-notebook-in-the-background-b6e950c4b7ee)

```
nohup jupyter lab &
```

to kill the process

```
lsof nohup.out
kill -9 <PID>
```