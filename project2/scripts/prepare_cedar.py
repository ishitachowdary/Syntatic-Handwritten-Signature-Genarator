import os, shutil

RAW_CEDAR_DIR = "raw_cedar"
TARGET_DIR = "data/cedar"
GENUINE_DIR = os.path.join(TARGET_DIR, "genuine")
FORGERY_DIR = os.path.join(TARGET_DIR, "forgery")

os.makedirs(GENUINE_DIR, exist_ok=True)
os.makedirs(FORGERY_DIR, exist_ok=True)

for folder in sorted(os.listdir(RAW_CEDAR_DIR)):
    if not folder.startswith("signatures_"):
        continue

    uid = int(folder.split("_")[1])
    user = f"user_{uid:02d}"

    src = os.path.join(RAW_CEDAR_DIR, folder)
    gdir = os.path.join(GENUINE_DIR, user)
    fdir = os.path.join(FORGERY_DIR, user)

    os.makedirs(gdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    for f in os.listdir(src):
        if not f.endswith('.png'): 
            continue
        if f.startswith('original'):
            shutil.copy(os.path.join(src,f), os.path.join(gdir,f))
        elif f.startswith('forgeries'):
            shutil.copy(os.path.join(src,f), os.path.join(fdir,f))

print("CEDAR prepared")