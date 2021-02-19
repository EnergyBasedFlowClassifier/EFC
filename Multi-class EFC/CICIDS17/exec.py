import zipfile

with zipfile.ZipFile("GeneratedLabelledFlows.zip", 'r') as zip_ref:
    zip_ref.extractall()

exec(open("join_web_attack.py").read())
print("join_web_attack Done")
exec(open("pre_process.py").read())
print("pre_process Done")
exec(open("drop_duplicates.py").read())
print("drop_duplicates Done")
exec(open("discretize.py").read())
print("discretize Done")
exec(open("encode_categorical.py").read())
print("encode_categorical Done")
exec(open("create_dir.py").read())
exec(open("5-folds.py").read())
print("5-folds Done")
exec(open("ajust_labels.py").read())
print("ajust_labels Done")
exec(open("undersampling.py").read())
print("undersampling Done")
