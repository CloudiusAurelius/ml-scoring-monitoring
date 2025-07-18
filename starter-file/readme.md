# 0. Configuration and Setup
- Course 3 / 1.10 Tools & Environment
    - Miniconda
        - https://www.anaconda.com/docs/getting-started/miniconda/install#mac-os
        - (see doc) Note that the package must be initialized by:
        ```Bash
        source ~/miniconda3/bin/activate
        ``` 
        and 
        ```Bash
        conda init --all
        ``` 
        - activates the **base** environment
    - Conda packages
    ```Bash
    conda create --name udacity python=3.8 mlflow jupyter pandas matplotlib requests -c conda-forge
    ```    
    
        - activate the **Udacity** environment

        ```Bash
        Downloading and Extracting Packages:
                                                                                                                                                                        
        Preparing transaction: done                                                                                                                                      
        Verifying transaction: done                                                                                                                                      
        Executing transaction: done                                                                                                                                      
        #                                                                                                                                                                
        # To activate this environment, use                                                                                                                              
        #                                                                                                                                                                
        #     $ conda activate udacity                                                                                                                                   
        #                                                                                                                                                                
        # To deactivate an active environment, use                                                                                                                       
        #                                                                                                                                                                
        #     $ conda deactivate   
        ```

    - Weights and Biases
        - account created with GitHub (au)
        - 7c191ceee0887771e6caf72d96abd63a733275b6
        - Python Package
            - with the **Udacity** environment activated (see above):
            ```Bash
            pip install wandb
            wandb login 
            ```
        - Test Installation
        ```Bash
        echo "wandb test" > wandb_test
        wandb artifact put -n testing/artifact_test wandb_test
        ```
    - Test MLflow
        ```Bash
        mflfow --help
        ```
    - Workflow: set up environments:
	    - fork a git repository (GitHub GUI --> fork)
	    - git clone repository to local
	    - create conda environment on local
        ```Bash		
        conda env create -f <environment_file>.yml #if a file is provided
        conda create -n <environment_name> "python=3.8" package1 package2 package 3...
        conda activate <environment_name>
        ``` 
    - Jupyter Lab
        - can be started with:
        ```Bash
        jupyter lab
        ```
