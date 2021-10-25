#Part 1#
import requests
import pandas as pd
import numpy as np
import os
get_resp = requests.get("http://3.85.131.173:8000/random_company")
get_resp.text
fake_html = get_resp.text.split("\n")
fake_html

n = 50
df = pd.DataFrame(index = np.arange(n), columns = ["Name", "Purpose"])
#filtering on Name and Purpose, 50 times
for i in range(n):
    get_resp = requests.get("http://3.85.131.173:8000/random_company")
    fake_html = get_resp.text.split("\n")
    for line in fake_html:
        if "Name" in line:
            a = line
        if "Purpose" in line:
            b = line
    df.iloc[i-1,:] = [a, b]

#getting rid of extra elements
df["Name"] = df["Name"].str.replace("</li>","")
df["Name"] = df["Name"].str.replace("<li>Name:","")
df["Purpose"] = df["Purpose"].str.replace("</li>","")
df["Purpose"] = df["Purpose"].str.replace("<li>Purpose:","")
print(df)

df.to_csv("fake_company.csv", sep='\t')

#####################################################
#Parts 2 and 3#
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
import os

os.chdir("/Users/hfsladmin/OneDrive - stevens.edu/PhD FE/Semester 3/FE 595- Fintech")
#os.getcwd()

# read the files
df1 = pd.read_csv('Companies.csv')
df2 = pd.read_csv('company_info.csv')
df3 = pd.read_csv('fake_company.csv')
df4 = pd.read_csv('name_purpose_pairs.csv')

# concat the dataframes
df = pd.concat([df1.rename(columns={'Company_Name':'Name', 'Company_Purpose':'Purpose'}) , df2 , df3 , df4.rename(columns={'Company_Name':'Name', 'Company_Purpose':'Purpose'})])
#df.shape()

# get the sentiment score
def sent_sc(text):
    sent = TextBlob(text).sentiment.polarity
    return sent

# add sentiment score to the df
df["sentiment"] = df.apply(lambda a: sent_sc(a["Purpose"]), axis=1)
#df.head()

# sort them based on sentiment score of purpose
df_sort = df.sort_values(by=["sentiment"])

# top and bottom 5
df_sort_worst = df_sort.head(5) #bottom
df_sort_best = df_sort.tail(5) #top

df_sort_worst.to_csv(r'company_worst.csv',  index=None, header=True, sep=';')
df_sort_best.to_csv(r'company_best.csv', index=None, header=True, sep=';')

# Observations
# most of the companies have sentiment score of 0. The lowest is -0.60 and the highest score is 0.50. This might be because of the way the column purpose is written; it's short and uses
# lots of technical words with neutrality which the pretrained sentiment analyzer might not take as polarity or might even not recognize the world. Also, this is a discription of a company
# which is pretty straightforward and objective.





