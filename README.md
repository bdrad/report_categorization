# Automatic Determination of Urgent Findings in Radiology Reports
Categorize reports based on whether or not they require follow-up. Live demo [here](http://urgent-findings-demo.herokuapp.com/index.html)

### Usage
The first step is to install fastText for Python: see instructions [here](https://github.com/facebookresearch/fastText#building-fasttext-for-python). Once you've done that we need to preprocess our data. We expect that our raw reports are in CSV files where the report text is in the "Report Text" column.
```
cd src
python preprocessing.py --sections impression -i /path/to/your/input.csv -o tmp_pp.bin
```
Next we apply the semantic mapping step; this could take a while depending on the size of your dataset 
```
python semantic_mapping.py tmp_pp.bin mapped_data.bin -s ../semantic_maps/clever_replacements ../semantic_maps/misc_replacements ../semantic_maps/radlex_replacements
```
Now we can use `src/fastTextClassify.ipynb` to run our fastText model. Set `labeled_reports_path` to be `mapped_data.bin` (or whatever you set the output path of the semantic mapping to be) and run the model!
