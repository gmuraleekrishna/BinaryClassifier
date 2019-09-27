# Dataset setup
The image files of this dataset have file names corresponding to its class (eg cat_xxx.tiff for cat image). To create a csv file with filenames and corresponsing labels for this  dataset run,
    

    $ cd <train_images_folder>
    
    $ ls | awk '/^[<class_name_1_in_file_name>]/ { printf("%s,%i\n", $0, <label_for_class_1>) } /^[<class_name_2_in_file_name>]/ { printf("%s,%i\n", $0, <label_for_class_2>) }' >> train_labels.csv

Usually the labels are chosen as 0 and 1 for binary classification. You can do the same in test images folder to create `test_label.csv`. Make sure to use same labels for the classes in both train and test csvs
