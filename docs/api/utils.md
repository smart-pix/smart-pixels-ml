# Utils Module

This module contains utility functions to manage file operations and GPU configurations.

## Functions

### `safe_remove_directory(directory_path)`
Safely removes a directory if it exists.

- **Arguments**:
  - `directory_path` (str): Path to the directory to be removed.
- **Example**:
  ```python
  from utils import safe_remove_directory
  safe_remove_directory("./temp_folder")


### `check_GPU()`
Checks for available GPUs and sets memory growth to prevent allocation issues.

- **Arguments**:
  - None.
- **Return**
  - Prints GPU information.
- **Example**:
  ```python
  from utils import safe_remove_directory
  safe_remove_directory("./temp_folder")