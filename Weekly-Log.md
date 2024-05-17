# Weekly Log

This is a weekly log of what work I have done on tasks for this research project. I have split the work up (roughly) by week, along with a separate section at the top for ongoing work/tasks.

## Current & Ongoing Tasks

- Theory-Related
    * [ ] Read Bayesian Optimization paper (<https://arxiv.org/pdf/2205.15449>)
    * [x] Brush up on linear algebra
    * [ ] Create and update theory/project notes document
- Coding-Related
    * [ ] Update the iterative GPs notebook to work properly with selection matrices 

---

## Completed Tasks (Organized by Week)

### May 12 - May 18

- [x] Learn/relearn the general necessary Python skills needed for this project
    * [x] Relearn general Python syntax and code structure
    * [x] Relearn pandas, numpy, scipy library syntax and common commands
    * [x] Learn how to use the Pytorch (torch) library
- [x] Make further changes to the [Iterative Gaussian Processes notebook](./Code/Demo-Code/iterative_gp_conditioning_updated.ipynb)
    * [x] Edit code to randomly generate "feature weight" matrices 
    * [x] Create a helper function (`normalize_selection`) to ensure the weight matrices have the same norm
    * [x] Write comments to explain the code
    * [ ] Write more detailed information in a separate cell with proper mathematical notation

### May 05 - May 11 

- [x] Create + organize Git repository for the research project
- [x] Create weekly log for task management
- [x] Complete STAT 520P problem set
- [x] Python setup
    * [x] Install and update Python
    * [x] Reinstall Anaconda and Jupyter
    * [x] Download common libraries (pandas, numpy, scipy, etc.)
- [x] Read additional Gaussian Process/Bayesian Optimization notes
- [x] Make changes to the [Iterative Gaussian Processes notebook](./Code/Demo-Code/iterative_gp_conditioning_demo_original.ipynb)
    * [x] Make the `update_K_chol` function more efficient
    * [x] Change the iterative selection in the loop in the final cell
