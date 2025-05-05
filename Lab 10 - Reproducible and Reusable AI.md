# Lab 10: Reproducible and Reusable AI

[COM6012 Scalable Machine Learning **2025**](https://github.com/COM6012/ScalableML) by Xianyuan Liu, and Lalu Muhammad Riza Rizky, 8th May 2025

## Study Schedule
- [Task 1](#1-Google-Colab): To finish in the lab session on 8th May. **Essential**
- [Task 2](#2-Containerised-ML-workflow-for-multi-site-autism-classification): To finish in the lab session on 8th May. **Essential**
- [Task 3](#3-Building-your-own-image): To explore further. *Optional*

### Suggested reading
- [Google Colab frequently asked questions](https://research.google.com/colaboratory/faq.html)
- [Introduction to Apptainer](https://apptainer.org/docs/user/main/introduction.html)

## ☁️ 1. Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xianyuanliu/alloy-property-extraction-demo/blob/main/NLP_for_Materials.ipynb)

## 📦 2. Containerised ML workflow for multi-site autism classification

### 💡 2.1. Introduction to containers for AI

Reproducing AI/ML experiments is often challenging due to a range of factors that can influence results, such as the operating system, programming language versions, library dependencies, and random seed settings.
Frequently, the original code is tightly coupled to a specific environment and setup, making it difficult to run elsewhere.
As a result, researchers often spend much time adapting code or setting up environments to replicate existing results — time that could be used to develop new models.


**[Containers](https://en.wikipedia.org/wiki/Containerization_(computing))** are a convenient tool for addressing the reproducibility issues common in AI/ML experiments. They allow practitioners to define and package a specific combination of environments, such as operating systems, libraries, and dependencies, ensuring that experiments can be reproduced exactly. Moreover, because the entire codebase is encapsulated within the container, it can be easily reused to run different experiments, saving time and effort.

This section explains how to deploy containers in a high-performance computing (HPC) environment.
Although [Docker](https://www.docker.com/) is the most widely used platform for container deployment, it is often not supported on HPC systems due to security concerns, as it requires root privileges.
Alternatives that do not require elevated access include [Apptainer](https://apptainer.org/) and [Podman](https://podman.io/).

In this guide, we will use **Apptainer**, as it is the container platform available on the University of Sheffield’s (UoS) HPC systems.

### 🔧 2.2. Setting up the container

Before deploying a container, we first need a container image that includes the environment and the code/program to be executed. 
> A _container image_ is a packaged snapshot of an environment, including the application code, libraries, and dependencies needed to run a program.
1. First, log in to the HPC using SSH. In this case, we are using the Stanage cluster.
    ```sh
    ssh $USER@stanage.shef.ac.uk
    ```
    Please replace `$USER` with your username (using **lowercase** and without `$`).

2. Once logged in, we can request a core from the general queue by
   ```sh
   srun --pty bash -i
   ```

3. Next, we pull and build the Apptainer container image from a registry. Commonly used registries include [Docker Hub](https://hub.docker.com) and [GitHub Container Registry (GHCR)](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry). In our case, we will pull an image from GHCR that is used to train and evaluate an autism classifier with a multi-site dataset called ABIDE.
   > A _registry_ is a storage and distribution system for these container images. 
   > Think of it like a version-controlled library or repository for containers. 
   > Developers build images locally and then push them to a registry so others can download (or pull) and run them in any compatible container runtime.

   In a real-world setting, the container image is typically pulled directly from a remote registry and built on the user's system. This is done using the following command:
   ```sh
     apptainer pull $IMAGE_NAME.sif docker://ghcr.io/zarizk7/abide-demo:master
   ```
   It will pull and build an image with a `*.sif` extension. You could replace `$IMAGE_NAME` with a name of your choice, such as `abide-demo`.
   If `$IMAGE_NAME` is left blank, by default it will be set to `$REPO_$TAG`, where `$REPO` is the repository name and `$TAG` is the image's version. 
   Once the image has been pulled, we can find it in our working/specified directory.

   However, as the image is quite large, pulling and building it can take up to 30 minutes. Therefore, this step is **optional** for the lab session.
   To save time, we provide a pre-built image on the HPC for you to use, which is stored at 
   ```
   /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif
   ```
   To use the pre-built image, we can skip the pull step and directly run the container.
   For the rest of the steps, we assume that the image is stored at this directory. 

### 🧐 2.3 Exploring the container

### ▶️️ 2.4. Running the container

1. After we pull and build the Apptainer image, we can deploy a container using the image to train and evaluate the model. With a container-/script-based code for model training/evaluation, there usually going to be many flags/variables that we can set. To see the available flags, we can call:
   ```sh
   apptainer run /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif -h
   ```
   We will find that there are many flags/variables that can be set to do experiment. The required ones are `--input-dir` and `--output-dir`, specifying the path to the dataset and output directory respectively.

2. Assuming that the dataset is in `/mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset`, to deploy the container for training and evaluation, we can run command:
    ```sh
    mkdir $HOME/outputs/abide-demo
    ```
   to create an output directory for the results, and then run:
   ```sh
   apptainer run \
        /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif \
        --input-dir /mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset \
        --output-dir $HOME/outputs/abide-demo \
        --random-state 0 \
        --verbose 1
   ```
   To ensure that the results obtained are reproducible, we will need to set an integer value for `--random-state`. Without it being set, we will not be able to get consistent results as some algorithms used for the model and evaluation is a stochastic method.

3. After the container finished running, the output directory will contain:
   - `args.yaml`: All of the arguments defined during the container's deployment time.
   - `cv_results.csv`: Cross-validation runtime, prediction scores, and hyperparameters.
   - `inputs.npz`: Features extracted from the data used to train the model.
   - `model.joblib`: A trained model using the optimal hyperparameter settings identified during the tuning process.
   - `phenotypes.csv`: Preprocessed phenotypic information of the subjects used for domain adaptation.

4. To check the results, we provide a python file `xx.py` to parse the scores from `cv_results.csv`. 
We can run it via 
   ```sh
   To be added
   ```
   and we should see the scores which is the same as we provided at the beginning of the lab session.

### 🧾 2.5. Running with `sbatch` (optional)
Provides the same functionality as in section 2.4, but implemented differently. 
Can be skipped if you are not interested in using `sbatch` to run the container.

Alternatively, we can use `sbatch` to run the Apptainer command to run the workflow. 
It is useful when we expect a long runtime as `srun` session will disconnect after certain time idling. 
Suppose that we already have the Apptainer image pulled (steps 1-3 done), we can create a shell file for sbatch to run. 
An example includes the following:

First, we create a shell script `run-abide-demo.sh` via `nano` or `vim`.

An example using `vim` is shown below.
```
cd $HOME
vim run-abide-demo.sh
```
Press `i` to enter the insert mode, and then copy and paste (`ctrl+shift+v`) the following code into the file.

```sh
#!/bin/bash

#SBATCH --job-name=abide-demo
#SBATCH --time=01:00:00

OUTPUT_DIR=$HOME/outputs/abide-demo

mkdir -p $OUTPUT_DIR

apptainer run \
    /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif \
    --input-dir /mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset \
    --output-dir $OUTPUT_DIR \
    --random-state 0
```
Then press `esc` to exit the insert mode, and type `:wq` to save and quit.

Next, deploy the container with `sbatch` we can simply call:
```sh
sbatch run-abide-demo.sh
```

To check the job's status, we can run:
```sh
sacct
```
which shows the progress of current/previous jobs. 
If it shows `RUNNING`, it means the job is still running. 

If we want to check the logs during the job's runtime, use command:
```sh
cat slurm-$JOB_NUMBER.out
```
where `$JOB_NUMBER` is the job number given when calling `sbatch`

Once the job is shown to be `COMPLETED` in `sacct`, we will expect the same output described in step 6.

### 🎲 2.6. Changing random seeds
To change the random seed, we can simply change the `--random-state` flag to a different integer value.
For example, to set the random seed to 1, we can run:
```sh
apptainer run \
    /mnt/parscratch/users/ac1xxliu/public/lab10-data/abide-demo.sif \
    --input-dir /mnt/parscratch/users/ac1xxliu/public/lab10-data/dataset \
    --output-dir $HOME/outputs/abide-demo \
    --random-state 1 \
    --verbose 1
```
The output will be saved in the same directory as before, but with a different random seed. 
    We can check the results by running the python script `xx.py` again to see the scores.
```sh
To be added python xx.py
```
The scores should be different from the previous run, as we have changed the random seed. 
This is because the random seed initialises the random number generator, which introduces controlled randomness into the training process of many machine learning models.

📌 Please do consider the random seed when you are running the model, as it is important for reproducibility.
Note that the random seed is not the only factor that can affect the reproducibility of the results. 
What other factors can affect the reproducibility of the results we talked in the lecture?

## 🛠️ 3. Building your own image