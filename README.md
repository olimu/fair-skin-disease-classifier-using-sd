# fair-skin-disease-classifier-using-sd
This study addresses the ethical implications of using neural networks for skin disease diagnosis. While such networks have demonstrated high accuracy, they tend to misdiagnose minority subpopulations as minorities are underserved and therefore underrepresented in current medical datasets. To prevent this, we balance a skin disease training dataset by adding stable diffusion-generated images depicting underrepresented dark skin toned patients with skin diseases. Additionally, prompt engineering and fine-tuning are utilized to optimize the realism and diversity of synthetic images. Notably, we achieve a statistically significant increase in the fairness of a skin disease detector, to our knowledge. Specifically, fairness increased by 58\%, as quantified by the average odds difference metric, helping to foster more equitable healthcare outcomes.

In the code folder:
- **_dreambooth_finetuning.ipynb_:** fine-tune diffusion model on Fitzpatrick17K images
- **_gen_training.py_:** preprocesses and creates the training dataset
- **_train_and_test.py_:** train and test VGG-16 networks to evaluate project 

In the presentations folder:
- **_ISEF_poster.pdf_:** trifold poster used to present at the International Science and Engineering Fair (ISEF) in 2024 
- **_quad_chart.png_:** quad chart summarizing the project
