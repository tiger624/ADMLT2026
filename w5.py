import os

desktop = r"C:\Users\c3698852\Desktop"

print("Desktop exists:", os.path.exists(desktop))

if os.path.exists(desktop):
    print("\nFiles on Desktop:")
    for file in os.listdir(desktop):
        print(file)