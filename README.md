# Bioacoustic ALAN project

This repo contain the code necessary to train a BirdNet custom classifier from a set annotations and the original file.
The script should be able to support different format, but currently only Raven, Audacity and Sonic Visualizer have been tested.

It is also possible to create a custom parser for even more data formats.
In order to do so, please follow the instruction inside the `parsers.py` script.

Possible commands:

```
usage: . [-h] {extract,train,validate} ...

Train and validate a custom BirdNet classifier based on given annotations by first exporting 3.0s segments.

positional arguments:
  {extract,train,validate}
    extract             Extracts audio chunks from long audio files using FFmpeg based on the given parser annotation. The result consists of       
                        multiple audio file, each 3s long "chunk", each in the corresponding labelled folder, which can be used to train the        
                        BirdNet custom classifier.
    train               Train a custom classifier using BirdNet Analyzer. The args are passed directly to `train.py` from BirdNet.
    validate            Validate the output from BirdNet Analyzer with some ground truth annotations. This creates two confusion matrices: one for  
                        the time and one for the count of (in)correctly identified segments of audio. From this, recall, precision and f1 score     
                        are computed and output in different tables.

options:
  -h, --help            show this help message and exit

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
```

the `-l` or `--label-settings` allow to easily manage different labels without the need to delete or rename folders.
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
Will match any label starting with "tyt", followed by any number of Unicode word characters.
Please note that `\w` has to be escaped into `\\w`.

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

Whitelist have precedence over blacklist.

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
```