import h5py
import os
import numpy as np

UKDALE_PATH = './dataset/ukdale'  # adjust relative path to your UK-DALE folder
OUTPUT_FILE = './dataset/ukdale/ukdale_tan.h5'

houses = ['house_1', 'house_2', 'house_5']  # only houses TAN-NILM uses

with h5py.File(OUTPUT_FILE, 'w') as f_out:
    for house in houses:
        house_path = os.path.join(UKDALE_PATH, house)
        if not os.path.exists(house_path):
            continue
        grp = f_out.create_group(house)
        # Process all channel_X files
        for filename in os.listdir(house_path):
            if filename.endswith('.dat') and 'channel' in filename:
                channel_path = os.path.join(house_path, filename)
                data = np.loadtxt(channel_path)
                grp.create_dataset(filename.replace('.dat',''), data=data)
        # Add mains dataset
        mains_path = os.path.join(house_path, 'mains.dat')
        if os.path.exists(mains_path):
            mains_data = np.loadtxt(mains_path)
            grp.create_dataset('mains', data=mains_data)
        else:
            print(f"[WARNING] mains.dat not found for {house}")
        print(f"Processed {house}")
