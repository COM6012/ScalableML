# Lab 10: Reproducible and Reusable AI

[COM6012 Scalable Machine Learning **2025**](https://github.com/COM6012/ScalableML) by ..., DD May 2025

## Study Schedule

### Suggested reading

## 1. Google Colab: 

## 2. Containerised ML workflow for multi-site autism classification

### 2.1. Introduction to containers for AI

Reproducing AI/ML experiments is often difficult because many things can affect the results, like the operating system, programming language version, or library versions used. Often, the original code only works in one specific setup. To get similar results, people might have to spend a lot of time changing the code or setting up the right environment. This takes time away from more useful work, like developing models, and instead forces practitioners to spend time just trying to reproduce existing results.

**Containers** are a convenient tool for addressing the reproducibility issues common in AI/ML experiments. They allow practitioners to define and package a specific combination of environments, such as operating systems, libraries, and dependencies, ensuring that experiments can be reproduced exactly. Moreover, because the entire codebase is encapsulated within the container, it can be easily reused to run different experiments, saving time and effort.

### 2.2. Setting up the container

Before deploying a container, we first need an image that includes the environment and the code/program to be executed. 

> The instructions in this section focus on how to deploy containers in a high-performance computing (HPC) environment. While Docker is the most widely used platform for container deployment, it is often not supported on HPC systems due to security concerns, as it requires root privileges. Alternatives that do not require root access include Apptainer and Podman. In this guide, we will use **Apptainer**, as it is the container platform available on the University of Sheffield’s (UoS) HPC systems.

1. First, log in to the HPC using SSH. In this case, we are using the UoS's Stanage HPC system.
    ```
    ssh $USER@stanage.shef.ac.uk
    ```
    Please replace `$USER` with your UoS account username and follow the instructions provided to gain access.

2. Once logged in, we need to request a worker node from the reserved resources by running:
    ```
    srun --account=rse-com6012 --reservation=rse-com6012-10 --cpus-per-task=2 --time=01:00:00 --pty /bin/bash
    ```
    The command requests a worker node for a one hour interactive bash session with two CPU cores reserved for the COM6012 course's lab 10.

3. Next, we need to pull and build the container image from a registry. Commonly used registries include [Docker Hub](https://hub.docker.com) and [GitHub Container Registry (GHCR)](https://ghcr.io). In our case, we will pull an image from GHCR that is used to train and evaluate an autism classifier with a multi-site dataset called ABIDE.
   > Given the time needed to pull and build the image, this step is considered **optional** for this lab session. The pre-built image can be found at `/mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif`.

    To pull and build the image, we can run:
    ```
    apptainer pull $IMAGE_NAME.sif docker://ghcr.io/zarizk7/abide-demo:master
    ```
    It will pull and build an image with a `*.sif` extension. If the `$IMAGE_NAME` is left blank, by default it will be set to `$REPO_$TAG`, where `$REPO` is the repository name and `$TAG` is the image's version. Once the image has been pulled, we can find it on our working/specified directory.
    
    For the rest of the steps, we assume that the image is stored at `/mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif`

4. After we pull and build the Apptainer image, we can deploy a container using the image to train and evaluate the model. With a container-/script-based code for model training/evaluation, there usually going to be many flags/variables that we can set. To see the available flags, we can call:
   ```
   apptainer run /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif -h
   ```
   We will find that there are many flags/variables that can be set to do experiment. The required ones are `--input-dir` and `--output-dir`, specifying the path to the dataset and output directory respectively.

5.  Assuming that the dataset is in `/mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset`, to deploy the container for training and evaluation, we can run command:
    ```
    apptainer run \
        /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif \
        --input-dir /mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset \
        --output-dir /users/$USER/abide-demo-out \
        --random-state 0
    ```
    To ensure that the results obtained are reproducible, we will need to set an integer value for `--random-state`. Without it being set, we will not be able to get consistent results as some algorithms used for the model and evaluation is a stochastic method.

    Optionally, to trace the container's process we may also add `--verbose 1` flag which will print the current step being run.

    > Remember to run `mkdir -p /users/$USER/abide-demo-out` first before deploying the container in case we have not created any directory. 

6. After the container finished running, the output directory will contain:
   - `args.yaml`: All of the arguments defined during the container's deployment time.
   - `cv_results.csv`: Cross-validation runtime, prediction scores, and hyperparameters.
   - `inputs.npz`: Features extracted from the data used to train the model.
   - `model.joblib`: A trained model using the optimal hyperparameter settings identified during the tuning process.
   - `phenotypes.csv`: Preprocessed phenotypic information of the subjects used for domain adaptation.
  
## 2.3. Using `sbatch` for executing code

Alternatively, we can use `sbatch` to run the Apptainer command to run the workflow. It is useful when we expect a long runtime as `srun` session will disconnect after certain time idling. Suppose that we already have the Apptainer image pulled (step 1-3 done), we can create a shell file for sbatch to run. An example include:

```
#!/bin/sh

mkdir -p /users/$USER/abide-demo-out

apptainer run \
    /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif \
    --input-dir /mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset \
    --output-dir /users/$USER/abide-demo-out \
    --random-state 0
```

Lets say the shell script name is `run-abide-demo.sh`. Before running the script with `sbatch`, we need to change the permission of the shell script to allow it to be executable by calling:

```
chmod +x run-abide-demo.sh
```


Then to run the container with `sbatch` we can simply call:
```
sbatch --account=rse-com6012 --reservation=rse-com6012-10 --cpus-per-task=2 --time=01:00:00 run-abide-demo.sh
```

## 3. Building your own image