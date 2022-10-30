import fileinput

meta = list(map(lambda x: x.strip(), fileinput.FileInput("vocab100.csv", openhook=fileinput.hook_encoded("utf-8", ''))))
print(meta)
