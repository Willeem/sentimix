#Translation is all you need

##How to recreate our experiments 

##Baseline 

The baseline is located in stacked_model. After installing the necessary packages, the main script can be ran by the following command:

```
python3 meta_classifier.py --files <trainfile> <testfile> --translated_files <translatedfile> <translated file> --save <model name>
```

##Normalisation
The scripts in the normalisation folder are used as follows. First run the following script:

```
python3 get_normalisation.py --files <trainfile> <validationfile> --language <language> --type init
```

This will result in two seperate files which can be used with the tool from Rob van der Goot (https://bitbucket.org/robvanderg/monoise/src/master/). After running this tool, the files are normalised. Then run the following command:

```
python3 get_normalisation.py --files <trainfile> <validationfile> --language <language> --type concatenate
```
This leaves you with a json file which can be used in our final systems. The translation was done by concatenating the sentences and pasting it in Google Spreadsheets. 

##Python notebooks
To run our python notebooks you need a certain structure in Google Drive. The structure we used was as follows:

```
├── My Drive
|   ├── Colab Notebooks
|   |   ├── shared_task
|   |   |   ├── han
|   |   |   ├── data
|   |   |   ├── xlmroberta
|   |   |   |   ├── results
|   |   |   |   ├── model_outputs
|   |   |   |   ├── model_cache
```
The data folder contains the normalised code mixed data in json format and two translated files for each code-mixed language. The format for naming the files is: <language>_<code-mixed language>_<train or dev>.txt. 