# Dataset setup
The image files of this dataset have file names corresponding to its class (eg cat_xxx.tiff for cat image). To create a csv file with filenames and corresponsing labels for this  dataset run,
    

    $ cd <train_data_folder>
    
    $ ls | awk '/^<class_name_to_match_in_the_file_names>/ {printf("%s,%d\n",$0, <label_for_class>) >> "train_labels.csv"} { printf("%s,%d\n",$0, <label_for_not_class>) >> 
    "train_labels.csv"}'

Usualy the labels are chosen as 0 and 1 for binary classification.
