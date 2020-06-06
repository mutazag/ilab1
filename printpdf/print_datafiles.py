#%%
from utils.config import Config

import pandas as pd

#%%

c = Config() 
c.validate_files()

#%%


df1 = pd.read_csv(c.uid_300K, nrows=1000)

#%%
# this code need to be refactored into a util
# or save a clean file to dataset 
df1_columns_index = [15, 2,3,5,7,8,9,10,11,12]
df1_columns_labels = ['uid','bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err']

# df1.index = df1.uid
# df1 = df1.iloc[:, df1_columns_index[:-1]]
# df1.columns = df1_columns_labels[:-1]

# re-order and label columns
df1 = df1.iloc[:, df1_columns_index]
df1.columns = df1_columns_labels

#%% 
df1.head()

#%%
# df1.reset_index(drop=True, inplace=True)
#%%[markdown]
# https://pbpython.com/pdf-reports.html
# jinja template: 
# ```
# <!DOCTYPE html>
# <html>
# <head lang="en">
#     <meta charset="UTF-8">
#     <title>{{ title }}</title>
# </head>
# <body>
#     <h2>Dataframe dump</h2>
#      {{ bilayer_ie_c33_table }}
# </body>
# </html>
# ```
#%%
def to_pdf(df, filename):
    # To populate those variable, we need to create a Jinja environment and get our template:
    from pathlib import Path
    print_dir = Path("printpdf")
    report_style_filename = print_dir / "report_style.css"
    report_template_filename = print_dir / "report_template.html"
    report_pdf_filename = print_dir / filename

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(report_template_filename.as_posix())

    template_vars = {
        "title": filename,
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
import time
timer = time.time()
to_pdf(df1, "300K_0-1000.pdf")
print(f"timer: {time.time() - timer}")
#%% 
# #%%
# df2 = pd.read_csv(c.uid_18M)# , nrows=100000)
# # %%

# df2_columns_index = [4,5,6,7,8,11,9,10,12,16]
# df2_columns_labels = ['bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err','uid (index)']

# df2.index = df2.uid
# df2 = df2.iloc[:,df2_columns_index[:-1]]
# df2.columns = df2_columns_labels[:-1]


#%%
