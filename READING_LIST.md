# Reading List & Annotated Bibliography

> A guided reading list for this project (data-leakage measurement in deep-learning
> lung-nodule detection on LUNA16). Papers are grouped by theme and tagged:
> **★ must-read** = core to *this* study; others are supporting/background.
> Bibliographic details (year, venue, DOI) were verified against the publishers.
> Use this file as the source for `THESIS.md` §12 (References).

---

## How to read these (suggested path)

1. Start with the **problem you're studying** → §A (data leakage & shortcut learning).
2. Then the **data you're using** → §B (LUNA16 / LIDC-IDRI).
3. Then **why it matters clinically** → §C (screening trials).
4. Then **what others built** → §D (DL for lung cancer).
5. Finally, **the methods you used** → §E (models) and §F (rigor).

**If you only read five:** Yagis 2021, Setio 2017, Geirhos 2020, Varoquaux & Cheplygina 2022, Ardila 2019.

---

## A. Data leakage & shortcut learning — the heart of this project ★

- **★ Yagis, E., Atnafu, S. W., García Seco de Herrera, A., et al. (2021).** "Effect of data
  leakage in brain MRI classification using 2D convolutional neural networks." *Scientific
  Reports* **11**, 22544. DOI: [10.1038/s41598-021-01681-w](https://doi.org/10.1038/s41598-021-01681-w)
  — **Your closest sibling.** Same experiment design on brain MRI: slice-level vs
  subject-level splits. Found slice-level leakage inflated test accuracy by **30–55%**. This is
  the paper to model your write-up on.

- **★ Geirhos, R., Jacobsen, J.-H., Michaelis, C., et al. (2020).** "Shortcut learning in deep
  neural networks." *Nature Machine Intelligence* **2**(11), 665–673. DOI:
  [10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z)
  — The framework for *why* a model learns an artifact (their toy example: a net keys on object
  *location*, not shape — exactly your center-vs-mediastinum bug). Cite when you discuss the bug.

- **★ Varoquaux, G. & Cheplygina, V. (2022).** "Machine learning for medical imaging:
  methodological failures and recommendations for the future." *npj Digital Medicine* **5**, 48.
  DOI: [10.1038/s41746-022-00592-y](https://doi.org/10.1038/s41746-022-00592-y)
  — Survey of how bias/leakage creep in at every pipeline step. Backbone of your limitations chapter.

- **Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012).** "Leakage in data mining:
  Formulation, detection, and avoidance." *ACM Transactions on Knowledge Discovery from Data
  (TKDD)* **6**(4), 1–21. DOI: [10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579)
  — The foundational, formal definition of data leakage. Cite for the *concept* itself.

- **Zech, J. R., Badgeley, M. A., Liu, M., et al. (2018).** "Variable generalization performance
  of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study."
  *PLOS Medicine* **15**(11), e1002683. DOI:
  [10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683)
  — The famous "the model learned the *hospital*, not the disease" result. Canonical confounding example.

- **DeGrave, A. J., Janizek, J. D. & Lee, S.-I. (2021).** "AI for radiographic COVID-19
  detection selects shortcuts over signal." *Nature Machine Intelligence* **3**, 610–619. DOI:
  [10.1038/s42256-021-00338-7](https://doi.org/10.1038/s42256-021-00338-7)
  — Models that look accurate but rely on confounders and fail in new hospitals. Pairs with Geirhos.

- **Roberts, M., Driggs, D., Thorpe, M., et al. (2021).** "Common pitfalls and recommendations
  for using machine learning to detect and prognosticate for COVID-19 using chest radiographs
  and CT scans." *Nature Machine Intelligence* **3**, 199–217. DOI:
  [10.1038/s42256-021-00307-0](https://doi.org/10.1038/s42256-021-00307-0)
  — Catalogs real-world leakage failure modes (incl. duplicate/overlapping data) across many studies.

## B. The dataset & challenge ★ (cite whenever you mention the data)

- **★ Setio, A. A. A., Traverso, A., de Bel, T., et al. (2017).** "Validation, comparison, and
  combination of algorithms for automatic detection of pulmonary nodules in computed tomography
  images: The LUNA16 challenge." *Medical Image Analysis* **42**, 1–13. DOI:
  [10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015)
  — *The* LUNA16 paper. Defines the 888-scan benchmark, the candidate set, and the FROC metric.
  Mandatory citation.

- **★ Armato, S. G. III, McLennan, G., Bidaut, L., et al. (2011).** "The Lung Image Database
  Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference
  database of lung nodules on CT scans." *Medical Physics* **38**(2), 915–931. DOI:
  [10.1118/1.3528204](https://doi.org/10.1118/1.3528204)
  — The underlying LIDC-IDRI dataset and the multi-radiologist annotation protocol your labels rest on.

## C. Clinical motivation — why CT screening matters

- **★ National Lung Screening Trial Research Team (2011).** "Reduced lung-cancer mortality with
  low-dose computed tomographic screening." *New England Journal of Medicine* **365**, 395–409.
  DOI: [10.1056/NEJMoa1102873](https://doi.org/10.1056/NEJMoa1102873)
  — The landmark RCT proving low-dose CT screening cuts lung-cancer mortality (~20%). Your §1 hook.

- **de Koning, H. J., van der Aalst, C. M., de Jong, P. A., et al. (2020).** "Reduced lung-cancer
  mortality with volume CT screening in a randomized trial" (NELSON). *New England Journal of
  Medicine* **382**, 503–513. DOI: [10.1056/NEJMoa1911793](https://doi.org/10.1056/NEJMoa1911793)
  — European confirmation of NLST, with volumetric nodule management.

## D. Deep learning for lung nodules / lung cancer

- **★ Ardila, D., Kiraly, A. P., Bharadwaj, S., et al. (2019).** "End-to-end lung cancer screening
  with three-dimensional deep learning on low-dose chest computed tomography." *Nature Medicine*
  **25**, 954–961. DOI: [10.1038/s41591-019-0447-x](https://doi.org/10.1038/s41591-019-0447-x)
  — The landmark 3D DL lung-cancer paper (Google), 94.4% AUC, matched/beat radiologists.

- **Setio, A. A. A., Ciompi, F., Litjens, G., et al. (2016).** "Pulmonary nodule detection in CT
  images: False positive reduction using multi-view convolutional networks." *IEEE Transactions
  on Medical Imaging* **35**(5), 1160–1169. DOI:
  [10.1109/TMI.2016.2536809](https://doi.org/10.1109/TMI.2016.2536809)
  — Classic candidate-classification approach on LIDC; relevant to how your candidates are used.

## E. The models you actually used (cite in §6)

- **He, K., Zhang, X., Ren, S. & Sun, J. (2016).** "Deep residual learning for image
  recognition." *CVPR*, 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  — **ResNet** (your `resnet50` backbone).

- **Liu, Z., Mao, H., Wu, C.-Y., et al. (2022).** "A ConvNet for the 2020s." *CVPR*, 11976–11986.
  DOI: [10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167)
  — **ConvNeXt** (your `convnext_tiny`).

- **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021).** "An image is worth 16×16 words:
  Transformers for image recognition at scale." *ICLR*. arXiv:
  [2010.11929](https://arxiv.org/abs/2010.11929)
  — **Vision Transformer (ViT)** (your `vit_base_patch16_224`).

- **Oquab, M., Darcet, T., Moutakanni, T., et al. (2024).** "DINOv2: Learning robust visual
  features without supervision." *Transactions on Machine Learning Research (TMLR)*. arXiv:
  [2304.07193](https://arxiv.org/abs/2304.07193)
  — **DINOv2** (your frozen linear-probe backbone).

## F. Methodology rigor (transfer learning + metrics)

- **★ Raghu, M., Zhang, C., Kleinberg, J. & Bengio, S. (2019).** "Transfusion: Understanding
  transfer learning for medical imaging." *NeurIPS 2019*, 3342–3352.
  [proceedings link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)
  — Asks whether ImageNet pretraining actually helps medical tasks (often less than assumed).
  Directly relevant to your `pretrained: true` choice — discuss it, don't just assume the benefit.

- **Reinke, A., Tizabi, M. D., Baumgartner, M., et al. (2024).** "Understanding metric-related
  pitfalls in image analysis validation" (Metrics Reloaded companion). *Nature Methods* **21**,
  182–194. DOI: [10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0)
  — Why AUROC / sensitivity / specificity, what they hide, and how class imbalance distorts them.

---

### Note on accuracy
All entries above were cross-checked against the publisher pages (Nature, NEJM, ACM, IEEE,
NeurIPS, arXiv). The four model/CVPR/ICLR/TMLR entries in §E use the canonical conference
versions; if your citation style requires a specific page range or proceedings volume, confirm
on the linked DOI/arXiv page.
