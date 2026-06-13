import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

scan_path = next(
    Path("/mnt/sfs/lung_cancer_detection/data/luna16/raw/subset0").rglob("*.mhd")
)

image = sitk.ReadImage(str(scan_path))
volume = sitk.GetArrayFromImage(image)

middle_slice = volume.shape[0] // 2


plt.imshow(volume[middle_slice], cmap="gray")
plt.axis("off")
plt.savefig("/mnt/sfs/lung_cancer_detection/reports/sample_slice.png")
