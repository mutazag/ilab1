
import pandas as pd
import pdfkit as pdf

import os 

print(os.getcwd())

import sys 

sys.path.append("..")
[print(p) for p in sys.path]



from os.path import dirname, join, abspath
print(dirname(__file__))
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from utils.config import Config

bilayer = ['Ta4Ni4Te8_Ta4Ni4Te8', 'Ta4Te10Pd6_Hf1N2', 'Ta2Ni4Te4_Ta4Ni4Te8',
           'Ta2Ni4Te2Se2_Ta4Ni4Te8', 'Ta4Te10Pd6_Ta4Te10Pd6', 'Ta4Ni4Te8_Ta2Ni4Te6']
mono1 = ['Ta4Ni4Te8', 'Ta4Te10Pd6', 'Ta2Ni4Te4',
         'Ta2Ni4Te2Se2', 'Ta4Te10Pd6', 'Ta4Ni4Te8']
mono2 = ['Ta4Ni4Te8', 'Hf1N2', 'Ta4Ni4Te8',
         'Ta4Ni4Te8', 'Ta4Te10Pd6', 'Ta2Ni4Te6']

df = pd.DataFrame({
    "bilayer": bilayer,
    "mono1": mono1,
    "mono2": mono2})
print(df)

df.to_html('./samples/test.html')
PdfFilename = './samples/pdfPrintOut.pdf'
pdf.from_file('./samples/test.html', PdfFilename)


c = Config()

df2 = pd.read_csv(c.predicted_300K)
df2.to_html('./samples/300K.html')
pdf.from_file('./samples/300K.html', './samples/300K.pdf')

print('END')
