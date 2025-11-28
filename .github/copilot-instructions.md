---
description: AI rules derived by SpecStory from the project AI interaction history
globs: *
---

## HEADERS

## PROJECT RULES

## CODING STANDARDS

## WORKFLOW & RELEASE RULES

## TECH STACK

*   Added `albumentations`
*   Added `opencv-python`
*   Added `timm`

## PROJECT DOCUMENTATION & CONTEXT SYSTEM

## DEBUGGING

*   **Shell Command Formatting**: When using `!` to run shell commands in Python Notebooks (e.g., Kaggle), ensure that newlines (`\`) are the last character on each line and there are no extra spaces after them. Alternatively, write the entire command on a single line to avoid indentation errors. Ensure that `!` commands are placed at the beginning of a line without any leading spaces, even within code blocks like `if` statements.

## GLOSSARY

### Data Loading
When working with custom datasets, especially on platforms like Kaggle, ensure the following:
1.  **Data Path Flexibility:** Hardcoded file paths must be avoided. Use relative paths or configurable paths via arguments.
2.  **Kaggle Input Directory:** Datasets on Kaggle are typically located in the `/kaggle/input/` directory. Code must be adapted to correctly reference this directory.
3.  **Custom Dataset Handling:** Implement flexible data loading mechanisms that can accommodate various dataset structures, potentially integrating an `ImageFolder` approach for flexibility. The `load_custom_dataset` function should be adaptable to handle a recursive traversal of client directories (e.g., `data/client_0/class_x/...`). The code should automatically detect the presence of `client_` prefixed folders, merge data from all clients into a single dataset, and perform an 80:20 train/test split.
4. **GPR Dataset Considerations**: When working with GPR (Ground Penetrating Radar) datasets, the data loading process should account for any specific preprocessing steps required, such as those outlined in external notebooks or documentation. Transformations should be applied consistently across training and validation sets, and dependencies should be minimized. The `GPR_ImageFolder` class should inherit from `torchvision.datasets.ImageFolder` and override the `__getitem__` method to seamlessly integrate the `albumentations` preprocessing. Transformations should match the notebook's augmentation strategy (e.g., including `A.ElasticTransform` and `A.CoarseDropout` when `args.aug_method == 1`). The GPR dataset should use OpenCV for image loading, support Albumentations for data augmentation, and resize images to 224x224 when using MobileViT models (including `mobilevit_s`), or 32x32 for CNN models. If the dataset lacks pre-defined train/test splits, ensure correct and separate transforms are applied. Utilize a transformed-free dataset and `CustomSubset` to prevent test set augmentation.
5.  **Train/Test Split Handling**: Ensure the data loading logic handles the train/test split appropriately. If the dataset is not already split into separate train/test folders, the loading mechanism should include a function to perform the split (e.g., an 8:2 split as mentioned in the interaction).
6. **Automatic Detection**: The code automatically detects client-prefixed folders (e.g., `client_0`, `client_1`, etc.).
7. **Data Merging**: If client folders are detected, the code automatically traverses these folders to merge data from all clients into a single dataset.
8. **Automatic Splitting**: The merged dataset is automatically split into 80% training and 20% testing sets.

### Argument Parsing
1.  **Data Directory Argument:** Add a `--data_dir` argument in `main.py` to specify the location of the custom dataset.
2. **Dataset Type Argument:** Add a `--dataset` argument in `main.py` to specify the dataset type, enabling the selection of custom datasets like `gpr_custom`. Ensure that when `gpr_custom` is selected, the `num_classes` parameter is automatically set to the correct number of classes (e.g., 8 for the GPR dataset).
3. **Augmentation Method Argument:** Integrate `args.aug_method` into `main.py` for argument handling, adapting the data augmentation strategy based on its value (e.g., setting `aug_method == 1` to include `A.ElasticTransform` and `A.CoarseDropout`).

### Server Logic
1.  **Passing Data Directory:** Ensure that the `data_dir` argument is correctly passed to the server logic, specifically to functions like `load_dataset` and `dataset_division`.
2.  **Model-Specific Image Resizing:** Update `dataset_division` in `serverBase.py` to dynamically adjust image sizes based on the selected model. For example, CNN models should use a resize value of 32, while MobileViT models (including `mobilevit_s`) and ResNet models should use 224. The default resize value should be 224 for models not explicitly listed.

### Output Paths
1.  **Kaggle Working Directory:** Ensure that output paths are directed to the `/kaggle/working/` directory, as this is the only writable directory on Kaggle.
2.  **Kaggle Notebook Environment:** Check for the Kaggle notebook environment and adjust output paths accordingly.

### Model Configuration
1. **Dynamic Output Layer**: Ensure model architectures, such as `CIFAR10Model` and `CIFAR100Model`, dynamically adapt their output layers based on the `num_classes` parameter to accommodate different datasets. The output layer should not be hardcoded to a specific number of classes.
2. **MobileViT Integration:** Add support for the `mobilevit` model (from the `timm` library) via a wrapper class in `MLModel.py`. Implement the `get_head_val` / `set_head_val` and `get_body_val` / `set_body_val` interfaces, separating the model's "body" (feature extractor) and "head" (classifier). The `--model` argument in `main.py` should support the `mobilevit` and `mobilevit_s` options, with the initialization logic loading the corresponding pretrained model from `timm`.

### General Configuration
1. **Data Distribution Control**: The `--non_iidtype` and `--alpha_dir` parameters are available to control the data distribution across clients, especially for simulation purposes. Non-IID data distribution types include pathological non-independent and identically distributed data (extreme data skew) and practical non-independent and identically distributed data (Dirichlet distribution or by class proportion).

### Parameter Tuning
1. **Batch Size**: If encountering out-of-memory errors, reduce the batch size (e.g., using the `--B` argument).
2. **Learning Rate**: For pre-trained models like MobileViT, a smaller learning rate (e.g., 0.001) may be more appropriate.

### Kaggle-Specific Instructions
1.  **Packaging:** Package the entire project directory into a `.zip` file. It is recommended to remove the `__pycache__` folder before compression.
2.  **Uploading Code**: Upload the `.zip` file to Kaggle as a new Dataset.
3.  **Uploading Data**: Upload the GPR image dataset to Kaggle as another Dataset.
4.  **Notebook Setup**: Create a new Kaggle Notebook. Attach both the code dataset and the image dataset to the Notebook.
5.  **Code Execution**: Copy the code from the Input directory to the Working directory in the notebook. Install the necessary dependencies.
6.  **Running Training**: Modify the `--data_dir` to point to the actual path of the image dataset in Kaggle.
7.  **NumPy Version Conflict:** If encountering `ValueError: numpy.dtype size changed`, it indicates a NumPy version conflict. Force uninstall the current NumPy and reinstall with a compatible version (numpy<2.0.0).
8.  **Shell Command Formatting**: When using `!` to run shell commands in Python Notebooks (e.g., Kaggle), ensure that newlines (`\`) are the last character on each line and there are no extra spaces after them. Alternatively, write the entire command on a single line to avoid indentation errors. Ensure that `!` commands are placed at the beginning of a line without any leading spaces, even within code blocks like `if` statements.