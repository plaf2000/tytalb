from tkinter import ttk, filedialog
from parsers import available_parsers
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
audio_dir_button = ttk.Button(audio_files_frame, text="Choose directory", command = input_command(audio_dir_name, filedialog.askdirectory))

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






window.mainloop()