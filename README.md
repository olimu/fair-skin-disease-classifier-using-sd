# fair-skin-disease-classifier-using-sd
This study addresses the ethical implications of using neural networks for skin disease diagnosis. While such networks have demonstrated high accuracy, they tend to misdiagnose minority subpopulations as minorities are underserved and therefore underrepresented in current medical datasets. To prevent this, we balance a skin disease training dataset by adding stable diffusion-generated images depicting underrepresented dark skin toned patients with skin diseases. Additionally, prompt engineering and fine-tuning are utilized to optimize the realism and diversity of synthetic images. Notably, we achieve a statistically significant increase in the fairness of a skin disease detector, to our knowledge. Specifically, fairness increased by 58\%, as quantified by the average odds difference metric, helping to foster more equitable healthcare outcomes.

To request access to the generated images, fill out this [google form](https://forms.gle/DZB8Uo5aC7XsBXcG9).

In the presentations folder:
- **_AP_Research_Paper.pdf_:** longer paper submitted to College Board for AP Research capstone
- **_Checklist.pdf_:** NeurIPS checklist of pre-print paper
- **_ISEF_poster.pdf_:** trifold poster used to present at the International Science and Engineering Fair (ISEF)
- **_pre_print.pdf_:** shorter pre-print paper submitted to NeurIPS call for high school papers
- **_quad_chart.png_:** quad chart summarizing the project

In the programs folder:
- **_dreambooth_finetuning.ipynb_:** fine-tune diffusion model on Fitzpatrick17K images
- **_gen_training.py_:** preprocesses and creates the training dataset
- **_train_and_test.py_:** train and test VGG-16 networks to evaluate project 


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
