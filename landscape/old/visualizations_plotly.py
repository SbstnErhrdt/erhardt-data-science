import sql
import pandas as pd
import plotly.express as px


import matplotlib
matplotlib.use('TkAgg')


sql.connect()

biontech = get_cpc.get_cpc(2,4)
df_biontech = pd.DataFrame(biontech, columns=['cpc', 'count'])


ax = df_biontech.plot.bar(x='cpc', y='count', rot=0)

fig = px.line_polar(df_biontech, r='count', theta='cpc', line_close=True)
fig.show()