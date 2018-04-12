import os
import plotly.plotly as py
import plotly.figure_factory as ff
import pandas as pd

file_name = "weather.xls"
script_dir = os.path.dirname(__file__)
rel_path = "documents/"+file_name
abs_file_path = os.path.join(script_dir, rel_path)
df = pd.read_csv(abs_file_path)

##rf = open(abs_file_path,"r");
##
##l = [[0 for x in range(2)] for y in range(13)]
##l = rf.read()
##print(l)
##data_matrix = [['Country', 'Year', 'Population'],
##               ['United States', 2000, 282200000],
##               ['Canada', 2000, 27790000],
##               ['United States', 2005, 295500000],
##               ['Canada', 2005, 32310000],
##               ['United States', 2010, 309000000],
##               ['Canada', 2010, 34000000]]
##
##table = ff.create_table(data_matrix)
##py.iplot(table, filename='simple_table')
