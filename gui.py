import os
from tkinter import ttk, filedialog
from parsers import available_parsers
from variables import BIRDNET_AUDIO_DURATION
import tkinter as tk

def input_command(var: tk.StringVar, filedialog_asker):
    def command():
        dir_path = filedialog_asker()
        var.set(dir_path)
    return command


window = tk.Tk()

window.title("Birdnet trainer and validator")
window.geometry("500x400")

tab_control = ttk.Notebook(window)

extract_tab = ttk.Frame(tab_control)
train_tab = ttk.Frame(tab_control)
validate_tab = ttk.Frame(tab_control)

tab_control.add(extract_tab, text="Extract")
tab_control.add(train_tab, text="Train")
tab_control.add(validate_tab, text="Validate")
tab_control.pack(expand = 1, fill ="both") 

# Tables directory
tables_dir_name = tk.StringVar()
annotations_frame = ttk.Frame(extract_tab)
tables_dir_label = ttk.Label(annotations_frame, text = "Annotations directory:")
tables_dir_chosen = ttk.Entry(annotations_frame, textvariable = tables_dir_name)
tables_dir_button = ttk.Button(annotations_frame, text="Select", command = input_command(tables_dir_name, filedialog.askdirectory))
tables_dir_label.pack(side="left")
tables_dir_chosen.pack(side="left")
tables_dir_button.pack(side="left")

# Recursive
recursive = tk.BooleanVar(annotations_frame, True)
recursive_button = ttk.Checkbutton(annotations_frame, variable=recursive)
recursive_text = ttk.Label(annotations_frame, text="Look recursively into directories")
recursive_button.pack(side="left")
recursive_text.pack(side="left")


annotations_frame.pack()


# Annotation format

annotations_format_frame = tk.Frame(extract_tab)
table_format = tk.StringVar(annotations_format_frame, "raven")
parser_strings = [p().names[0] for p in available_parsers]
formats_menu = tk.OptionMenu(annotations_format_frame, table_format, *parser_strings)
formats_menu.pack(side="left")

# Annotation header
header = tk.BooleanVar(annotations_format_frame, True)
header_button = ttk.Checkbutton(annotations_format_frame, variable=header)
header_text = ttk.Label(annotations_format_frame, text="Annotation tables have header.")

header_button.pack(side="left")
header_text.pack(side="left")
annotations_format_frame.pack(pady=(10,0))

# Audio files directory
audio_files_frame = tk.Frame(extract_tab)
audio_dir_name = tk.StringVar(audio_files_frame)
audio_dir_label = ttk.Label(audio_files_frame, text = "Audio files directory:")
audio_dir_chosen = ttk.Entry(audio_files_frame, textvariable = audio_dir_name)

def audio_dir_command():
    input_command(audio_dir_name, filedialog.askdirectory)()
    labels_path = os.path.join(audio_dir_name.get(), "labels.json")
    if os.path.isfile(labels_path):
        label_settings.set(labels_path)

audio_dir_button = ttk.Button(audio_files_frame, text="Select", command = audio_dir_command)

audio_dir_label.pack(side="left")
audio_dir_chosen.pack(side="left")
audio_dir_button.pack(side="left")
audio_files_frame.pack(pady=(10,0))

# Output directory
out_dir_frame = tk.Frame(extract_tab)
out_dir = tk.StringVar(out_dir_frame)
out_dir_label = ttk.Label(out_dir_frame, text = "Output directory:")
out_dir_chosen = ttk.Entry(out_dir_frame, textvariable = out_dir)
out_dir_button = ttk.Button(out_dir_frame, text="Select", command = input_command(out_dir, filedialog.askdirectory))

out_dir_label.pack(side="left")
out_dir_chosen.pack(side="left")
out_dir_button.pack(side="left")
out_dir_frame.pack(pady=(10,0))

# Labels settings
label_settings_frame = tk.Frame(extract_tab)
label_settings = tk.StringVar(label_settings_frame)
label_settings_label = ttk.Label(label_settings_frame, text = "Label settings file:")
label_settings_chosen = ttk.Entry(label_settings_frame, textvariable = label_settings)
label_settings_button = ttk.Button(label_settings_frame, text="Select", command = input_command(label_settings, filedialog.askopenfilename))

label_settings_label.pack(side="left")
label_settings_chosen.pack(side="left")
label_settings_button.pack(side="left")
label_settings_frame.pack(pady=(10,0))

#Chunk overlap
def validate_overlap(P, V) -> bool:
    if P=="":
        if V=="focusout":
            P = 0.0
            chunk_overlap.set(P)
        return True
    try:
        P = float(P)
    except ValueError:
        return False
    return  0 <= P < BIRDNET_AUDIO_DURATION


chunk_overlap_frame = tk.Frame(extract_tab)
vcmd = chunk_overlap_frame.register(validate_overlap)
chunk_overlap = tk.DoubleVar(chunk_overlap_frame)
chunk_overlap_entry = ttk.Entry(chunk_overlap_frame, validate="all", textvariable=chunk_overlap, validatecommand=(vcmd, "%P", "%V"))

chunk_overlap_frame.pack()
chunk_overlap_entry.pack()

# Date format (input audio files)


# Late start

# Early stop

# Include path

# Stats only

# Noise export ratio

window.mainloop()