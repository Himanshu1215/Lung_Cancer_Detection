import SimpleITK as sitk
from pathlib import Path

scan_path = next(
    Path("/mnt/sfs/lung_cancer_detection/data/luna16/raw/subset0/subset0").glob("*.mhd")
)

print("Loading:", scan_path.name)

image = sitk.ReadImage(str(scan_path))
volume = sitk.GetArrayFromImage(image)

print("\nShape:")
print(volume.shape)

print("\nSpacing:")
print(image.GetSpacing())

print("\nOrigin:")
print(image.GetOrigin())

print("\nHU Range:")
print(volume.min(), volume.max())