#%%
from utils.config import Config

import pandas as pd

#%%


def to_pdf(df, range_str, filename):
    # To populate those variable, we need to create a Jinja environment and get our template:
    from pathlib import Path
    print_dir = Path("printpdf")
    report_style_filename = print_dir / "report_style.css"
    report_template_filename = print_dir / "report_template.html"

    filename = Path(filename)
    newfilename = f"{filename.stem}-{range_str}{filename.suffix}" 
    report_pdf_filename = print_dir / newfilename

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(report_template_filename.as_posix())

    template_vars = { 
        "title": newfilename, 
        "bilayer_ie_c33_table": df.to_html()}

    html_out = template.render(template_vars)

    # generate pdf from rendered html 

    from weasyprint import HTML, CSS
    report_style = CSS(filename=report_style_filename)
    # output to pdf
    HTML(string=html_out).write_pdf(
        report_pdf_filename,
        stylesheets=[report_style])
#%%

c = Config() 
df1_columns_index = [15, 2,3,5,7,8,9,10,11,12]
df1_columns_labels = ['uid','bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err']



#%%

import time
chunk_size = 5000
for i, df_chunk in enumerate(pd.read_csv(c.uid_300K, chunksize=chunk_size)):
    # re-order and label columns
    df_chunk = df_chunk.iloc[:, df1_columns_index]
    df_chunk.columns = df1_columns_labels
    range_str = f"{i*chunk_size} to {(i*chunk_size)+df_chunk.shape[0]-1}"
    print(range_str)
    timer = time.time()
    to_pdf(df_chunk, range_str, "300K.pdf")
    print(f"timer: {time.time() - timer}")

#%% [markdown]
# # print 18M



#%% 

df2_columns_index = [15,4,5,6,7,8,11,9,10,12]
df2_columns_labels = ['uid','bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err']

chunk_size = 10000
for i, df_chunk in enumerate(pd.read_csv(c.uid_18M, chunksize=chunk_size)):
    # re-order and label columns
    df_chunk = df_chunk.iloc[:, df2_columns_index]
    df_chunk.columns = df2_columns_labels
    range_str = f"{i*chunk_size} to {(i*chunk_size)+df_chunk.shape[0]-1}"
    print(range_str)
    timer = time.time()
    to_pdf(df_chunk, range_str, "18M.pdf")
    print(f"timer: {time.time() - timer}")


#%%
