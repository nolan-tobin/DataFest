The code in this repository is a bit cobbled together - for DataFest we only had 48 hours to analyze data, come up with a question/research
topic, write and test code, and create a presentation. Since my team and I were in a time crunch we didn't spend a lot of extra time optimizing
and polishing. It might be useful to read this page from the DataFest website. https://ww2.amstat.org/education/datafest/

The American Bar Association was the data sponsor for DataFest 2023. The data consisted of very large .xlsx files, one of which consited of 
nearly half a million rows which held questions submitted by clients and a category the query was associated with. The categories were as 
follows: Consumer Financial Questions, Family & Children, Health and Disability, Housing & Homelessness, Income Maintenance, Individual
Rights, Juvenile, Work Employment Unemployment, and Other. My team decided to filter out certain words such as "the", "a", "and", etc. in
order to speed up runtime.

Our goal was to steamline the process of sorting queries into the correct categories. To do this, my team created a recurring neural 
network model to sort new queries into these categories with accuracies around 93% throughout test runs. We achieved this by creating
training sets from 80% of the given data and testing sets from the remaining 20%. We defined our architecture of our recurring neural 
network using the Keras functional API, including an input layer, an embedding layer, a simple RNN layer, and a dense output layer. The 
model takes a question and returns, in order, the top 3 most likely categories, as well as the percentage it assigns to each.
