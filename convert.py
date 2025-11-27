from nilmtk import DataSet
from nilmtk.datastore import HDFDataStore

ukdale = DataSet("dataset/ukdale/ukdale.h5")  # path to your h5 or raw folder
ukdale.set_window(start="2013-04-12", end="2013-06-01")  # optional

# Save to new HDF5 in TAN format
output_path = "dataset/ukdale/ukdale_tan.h5"
store = HDFDataStore(output_path, 'w')
for house in ukdale.buildings:
    for elec in ukdale.buildings[house].elec:
        store.put(f"house_{house}", elec)
store.close()
print("Converted UK-DALE to TAN-compatible HDF5")
