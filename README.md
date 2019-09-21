To create filename and labels file for a binary dataset where the file names are the class names, run
in the directory


    ls | awk '/^<fill_class_label_here>/ {printf("%s,%d\n",$0, 1) >> "train_labels.csv"} { printf("%s,%d\n",$0, 0) >> 
    "train_labels.csv"}'
