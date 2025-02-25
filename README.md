# COM6012 Scalable Machine Learning - University of Sheffield

## Spring 2025

**by [Shuo Zhou](https://shuo-zhou.github.io/) and [Haiping Lu](https://haipinglu.github.io/), with [Tahsin Khan](https://www.sheffield.ac.uk/dcs/people/academic/tahsinur-khan) and [Xianyuan Liu](https://xianyuanliu.github.io/)**

In [this module](http://www.dcs.shef.ac.uk/intranet/teaching/public/modules/msc/com6012.html), we will learn how to do machine learning at large scale using [Apache Spark](https://spark.apache.org/).
We will use the [High Performance Computing (HPC) cluster systems](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html) of our university. To access the HPC clusters, log in using SSH with your university username and the associated password. When connecting while on campus using Eduroam or off campus, you **must** keep the [university's VPN](https://www.sheffield.ac.uk/it-services/vpn) connected all the time. Multifactor authentication (MFA) will be mandatory. The standard University [DUO MFA](https://www.sheffield.ac.uk/it-services/mfa/set-mfa#duo) is utilized.

This edition uses [**PySpark 3.5.4**](https://spark.apache.org/docs/3.5.4/api/python/index.html), the [latest stable release of Spark](https://spark.apache.org/releases/spark-release-3-5-4.html) (Dec 20, 2024), and has 10 sessions below. You can refer to the [overview slides](https://github.com/COM6012/ScalableML/blob/master/Slides/Overview-COM6012-2025.pdf) for more information, e.g. timetable and assessment information.

* Session 1: Introduction to Spark and HPC (Shuo Zhou)
* Session 2: RDD, DataFrame, ML pipeline, & parallelization (Shuo Zhou)
* Session 3: Scalable logistic regression and Spark configuration (Shuo Zhou)
* Session 4: Scalable generalized linear models and Spark data types (Shuo Zhou)
* Session 5: Scalable decision trees and ensemble models (Tahsin Khan)
* Session 6: Scalable neural networks (Tahsin Khan)
* Session 7: Scalable k-means clustering (Tahsin Khan)
* Session 8: Scalable matrix factorization for collaborative filtering in recommender systems and PCA for dimensionality reduction (Haiping Lu)
* Session 9: Apache Spark in the Cloud (Xianyuan Liu)
* Session 10: Reproducible and reusable AI  (Xianyuan Liu)

You can also download the [Spring 2024 version](https://github.com/COM6012/ScalableML/releases/tag/v2024) for preview or reference.

If you do not have a [GitHub account](https://github.com/join) yet, we recommend signing up for one to learn how to use this popular open-source software development platform.

We use US spelling in the slides and lab notes for consistency with the naming conventions in Spark.

## An Introduction to Transparent Machine Learning

Shuo Zhou and Haiping Lu developed a course on [An Introduction to Transparent Machine Learning](https://pykale.github.io/transparentML/) with [Prof. Haiping Lu](https://haipinglu.github.io/), part of the [Alan Turing Institute’s online learning courses in responsible AI](https://www.turing.ac.uk/funding-call-online-learning-courses-responsible-ai). If interested, you can refer to this introductory course with emphasis on transparency in machine learning to assist you in your learning of scalable machine learning.

## Acknowledgement

The materials are built with references to the following sources:

* The official [Apache Spark documentations](https://spark.apache.org/). *Note: the **latest information** is here.*
* The [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/) by [Wenqiang Feng](https://www.linkedin.com/in/wenqiang-feng-ph-d-51a93742/) with [PDF - Learning Apache Spark with Python](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf). Also see [GitHub Project Page](https://github.com/runawayhorse001/LearningApacheSpark). *Note: last update in Dec 2022.*
* The [**Introduction to Apache Spark** course by A. D. Joseph, University of California, Berkeley](https://www.mooc-list.com/course/introduction-apache-spark-edx). *Note: archived.*
* The book [Learning Spark: Lightning-Fast Data Analytics](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/), 2nd Edition, O'Reilly by Jules S. Damji, Brooke Wenig, Tathagata Das & Denny Lee, with a [github repository](https://github.com/databricks/LearningSparkV2).
* The book [**Spark: The Definitive Guide**](https://books.google.co.uk/books/about/Spark.html?id=urjpAQAACAAJ&redir_esc=y) by Bill Chambers and Matei Zaharia. There is also a Repository for [code](https://github.com/databricks/Spark-The-Definitive-Guide) from the book.

Many thanks to

* [Robert Loftin](https://www.sheffield.ac.uk/cs/people/academic/robert-loftin) and [Mauricio A Álvarez](https://maalvarezl.github.io/), who contributed to this module in 2024 and from 2016 to 2022, respectively. Their contributions remain reflected in the course materials.
* Mike Croucher, Neil Lawrence, William Furnass, Twin Karmakharm, Mike Smith, Xianyuan Liu, Desmond Ryan, Steve Kirk, James Moore, and Vamsi Sai Turlapati for their inputs and inspirations since 2016.
* Our teaching assistants and students who have contributed in many ways since 2017.
