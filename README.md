**Transitioning from Data Scientist to Machine Learning Engineer**

**Why?**
I have primarily worked as a Data Scientist in my career. This involves performing various tasks along the entire data pipeline (data engineer, data science, and machine learning) to help companies leverage their data for insight or products. I always felt like I was in some sort of protective bubble by using the typical data science workflow (Cloud jupyter notebooks like Databricks or Vertex AI Workbenches, storage buckets set up in different workspaces to hold data, and large data tables set up via data factories or BigQuery). Machine Learning (and Artificial Intelligence in general) seems like the natural step forward for someone who wants to leverage data for real value.

**What's the difference?**
The duties and skills of a Data Scientist and Machine Learning Engineer differ enough to require a serious amount of training. While my work as a Data Scientist uses similar tools as Machine Learning Engineers (Python, Git, Scikit-learn, PyTorch, PySpark, Polars/Pandas, and SQL), the standards for using them are different. Data Scientists can get away with running inefficient code since usually the purpose of their applications is data insights run just for internal use. Machine Learning Engineers, on the other hand, have to write production code used by customers. It needs to follow industry best practices and perform. While I am very confident in my knowledge of statistics and artificial intelligence algorithms, my implementation skills have atrophied since becoming a Data Scientist. Thus, I need to get to work.

**What is this repo?**
This repository is both a log and storage space to show off what I've learned on my journey to become a machine learning engineer. You can click through the various sub-projects to see what I did and what I learned. I also include summaries in this README. All projects were developed in Neovim and the command line except the cloud-based ones.

**ds_project**
After reading the insights in "Hitchhiker's Guide to Python", I wanted to use the info to make a standard project using industry tools I've never used before. Productions tools like MyPy, PyLint, Black, .toml files, and MakeFiles are foreign to me as a Data Scientist who primarily worked in notebook based environments. I had Claude Code create a basic Data Science project then add random typing and project structure errors so I would be FORCED to used these tools to find the issues and resolve them. After a little struggle with understanding MyPy errors, I managed to resolve the errors, run the project, and ensure it works as intended. Using PyTest is fairly simple for this project, but it's nice actually use it.
TECH LEARNED: MyPy, PyLint, Black, .toml files, MakeFiles, PyTest

