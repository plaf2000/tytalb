# Bioacoustic ALAN project

This repo contain the code necessary to train a BirdNet custom classifier from a set annotations and the original file.
The script should be able to support different format, but currently only Raven, Audacity and Sonic Visualizer have been tested.

It is also possible to create a custom parser for even more data formats.
In order to do so, please follow the instruction inside the `parsers.py` script.

Possible commands:

```
Train and validate a custom BirdNet classifier based on given annotations by first exporting 3.0s segments.

positional arguments:
  {extract,correct,train,validate}
    extract             Extracts audio chunks from long audio files using FFmpeg based on the given parser annotation. The result consists of multiple audio       
                        files ("chunks"), each 3s long, placed in the corresponding labelled folder, which can be used to train the BirdNet custom classifier.     
    correct             Correct the labels based on the mappings in a json file.
    train               Train a custom classifier using BirdNet Analyzer. The args are passed directly to train.py from BirdNet.
    validate            Validate the output from BirdNet Analyzer with some ground truth annotations. This creates two confusion matrices: one for the time        
                        (confusion_matrix_time.csv) and one for the count (confusion_matrix_count.csv) of (in)correctly identified segments of audio. From this,    
                        recall, precision and f scores are computed and output in different tables (validation_metrics_count.csv and
                        validation_metrics_time.csv).

options:
  -h, --help            show this help message and exit
```

## Requirements

FFmpeg has to be installed and accessible with the command `ffmpeg`.
Please find the other requirements in "requirements.txt":
```
pip install -r requirements.txt
```

## Extract

```
usage: . extract [-h] [-i TABLES_DIR] [-re RECURSIVE] -f {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}
                 [-a AUDIO_FILES_DIR] -ie AUDIO_INPUT_EXT [-oe AUDIO_OUTPUT_EXT] [-o EXPORT_DIR] [-l LABEL_SETTINGS_PATH] [-r RESAMPLE]
                 [-co CHUNK_OVERLAP] [-df DATE_FORMAT]

options:
  -h, --help            show this help message and exit
  -i TABLES_DIR, --input-dir TABLES_DIR
                        Path to the folder of the (manual) annotations.
  -re RECURSIVE, --recursive RECURSIVE
                        Wether to look for tables inside the root directory recursively or not (default=True).
  -f {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}, --annotation-format {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}
                        Annotation format.
  --header, --no-header
                        Whether the annotation tables have an header. The default value is defined by the annotations parser.
  -a AUDIO_FILES_DIR, --audio-root-dir AUDIO_FILES_DIR
                        Path to the root directory of the audio files (default=current working dir).
  -ie AUDIO_INPUT_EXT, --audio-input-ext AUDIO_INPUT_EXT
                        Key-sensitive extension of the input audio files (default=wav).
  -oe AUDIO_OUTPUT_EXT, --audio-output-ext AUDIO_OUTPUT_EXT
                        Key-sensitive extension of the output audio files (default=flac).
  -o EXPORT_DIR, --output-dir EXPORT_DIR
                        Path to the output directory. If doesn't exist, it will be created.
  -l LABEL_SETTINGS_PATH, --label-settings LABEL_SETTINGS_PATH
                        Path to the file used to map and filter labels. Please refer to `README.md`. By default the file is `labels.json` in the    
                        root directory of annotations.
  -r RESAMPLE, --resample RESAMPLE
                        Resample the chunk to the given value in Hz. (default=48000)
  -co CHUNK_OVERLAP, --chunk-overlap CHUNK_OVERLAP
                        Overlap in seconds between chunks for segments longer than 3.0s. If it is 0 (by default) the program may run faster.
  -df DATE_FORMAT, --date-fromat DATE_FORMAT
                        Date format of the file. (default = "%Y%m%d_%H%M%S")
  -ls LATE_START, --late-start LATE_START
                        Whether to not consider the interval between the start of the recording and the first annotation (default = False)
  -es EARLY_STOP, --early-stop EARLY_STOP
                        Whether to not consider the interval between the last annotation and the end of the recording (default = False)
  -ip, --include-path, --no-include-path
                        Whether to include the relative path in the output file name (default = False). If two filenames are not unique, this will  
                        be done automatically.
```
## Label mapping/correction

the `-l` or `--label-settings` allow to easily manage different labels without the need to delete or rename folders or move their contents, just by changing the settings on the corresponding json file.

The mapping to new labels is done following this procedure:
1. Cleaning
    1. Strip
    1. Single whitespace
    1. Lowercase
1. Substitute
1. Map
1. Black/whitelist

**Be aware of the order** of these operations!

### Cleaning

This operations are meant to make the labels simpler and avoid some common typos.

#### Strip
This operation removes white spaces at the beginning and the end of the label - using the `strip()` Python method, since this can be a common typo.

This is **always performed by default**. To avoid this behaviour, set the `"strip"` attribute to `false` in the json file.

#### Single whitespace
This operation replaces multiple white spaces, including tabs and other special white space characters, in the label, since this can be a common typo.

This is **always performed by default**. To avoid this behaviour, set the `"single whitespace"` attribute to `false` in the json file.

#### Lowercase
This operation turns the label into lowercase, to make the labels case insensitive and avoid case-typos.

This is **not performed by default**. To set this behaviour, set the `"lower"` attribute to `true` in the json file.

### Substitute

You can substitute portions of labels matching some regex by using the `"sub"` attribute in the json settings. For example
```json
{
    "sub": {
        "alb\\w*": "alb"
    }
}
```

will substitute any portion of labels starting containing "alb", followed by any number of Unicode word characters with just "alb", so for instance "tytalbalb" and "tytalb1" will become simply "tytalb".

To keep parts of the original label use the regex group and include the group number in the substitution string. For example

```json
{
    "sub" : {
        "(\\w+),(\\w+)": "\\1, \\2"
    }
}

```
will turn "tytalb,Barn Owl" to "tytalb, Barn Owl".


**Note:** backslashes have to be escaped, so in this case `\w` becomes `\\w`.

### Map

You can use the `"map"` attribute to map the labels you have inside the provided tables to new labels.
For example
```json
{
    "map": {
        "tytalb": "Tyto alba_Barn Owl"
    }
}
```
will automatically rename all "tytalb" labels to "Tyto alba_Barn Owl". It is also possible to use Python regex to 
match the label pattern. For example
```json
{
    "map": {
        "tyt\\w*": "Tyto"
    }
}
```
will match any label starting with "tyt", followed by any number of Unicode word characters.



### White/blacklist
It is also possible to whitelist or blacklist labels using the corresponding
attributes. Note that the black-/whitelisting will be applied once the label 
is already mapped to the new one.
For example
```json
{
    "map": {
        "tyt\\w*": "Tyto"
    },
    "blacklist": ["Tyto"]
}
```
will ignore all labels matching `r"tyt\w*"`. 
```json
{
    "map": {
        "tyt\\w*": "Tyto"
    },
    "whitelist": ["Tyto"]
}
```
will instead consider all the other labels as noise.

**Note:** Whitelist have precedence over blacklist.

### Example
```json
{
    "map": {
        "TURPHI": "Song thrush, Turdus philomelos",
        "TRIOCH": "Green sandpiper, Tringa ochropus",
        "EMBHOR": "Ortolan bunting, Emberiza hortulana",
        "ANTTRI": "Tree pipit, Anthus trivialis",
        "CORVID": "Corvidae",
        "NotNocMig PICVIR": "Green woodpecker, Picus viridis, notnocmig",
        "IXOMIN": "Little bittern, Ixobrychus minutus",
        "TURILI": "Redwing, Turdus iliacus",
        "GALCHL": "Moorhen, Gallinula chloropus",
    },
    "sub": {
        "'": "",
        "(\\w+),(\\w+)": "\\1, \\2"
    },
    "lower": true
}
```

## Validate

```
usage: . validate [-h] [-gt TABLES_DIR_GT] [-tv TABLES_DIR_TV] -fgt {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}
                  [-ftv {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}] [-o OUTPUT_DIR] [-re RECURSIVE]

options:
  -h, --help            show this help message and exit
  -gt TABLES_DIR_GT, --ground-truth TABLES_DIR_GT
                        Path to the folder of the ground truth annotations (default=current working dir).
  -tv TABLES_DIR_TV, --to-validate TABLES_DIR_TV
                        Path to the folder of the annotations to validate (default=current working dir).
  -fgt {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}, --annotation-format-ground-truth {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}
                        Annotation format for ground truth data.
  -ftv {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}, --annotation-format-to-validate {sonic-visualizer,sv,audacity,ac,raven,rvn,kaleidoscope,ks,birdnet_raven,bnrv}
                        Annotation format for data to validate (default=raven).
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to the output directory (default=current working dir).
  -re RECURSIVE, --recursive RECURSIVE
                        Wether to look for tables inside the root directory recursively or not (default=True).
  -ls, --late-start, --no-late-start
                        Whether to not consider the interval between the start of the ground truth recording and the first annotation (default =    
                        False)
  -es, --early-stop, --no-early-stop
                        Whether to not consider the interval between the last annotation and the end of the recording (default = False)
  -b, --binary, --no-binary
                        Whether to validate as binary classification. If set, and the POSITIVE_LABEL is not provided, an exception will be raised.  
  -p POSITIVE_LABELS, --positive-labels POSITIVE_LABELS
                        Comma-separated labels considered as positive for the binary classification.
  -cts CONFIDENCE_THRESHOLDS_START, --conf-thresholds-start CONFIDENCE_THRESHOLDS_START
                        Start range for confidence thresholds.
  -cte CONFIDENCE_THRESHOLDS_END, --conf-thresholds-end CONFIDENCE_THRESHOLDS_END
                        End range for confidence thresholds
  -ct CONFIDENCE_THRESHOLDS, --conf-thresholds CONFIDENCE_THRESHOLDS
                        Number of thresholds to filter the data to validate (linearly distributed between CONFIDENCE_THRESHOLDS_START and
                        CONFIDENCE_THRESHOLDS_END). The table format must have a field for the confidence and it has to be defined in the parser.
```

## Problems
- If ground-truth annotation times are overlapping, in the matrix they will appear multiple times, so some annotations can be considered wrong although correct.
- The true negatives (noise matching noise) are not registered in the matrix.