# Weekly Log

This is a weekly log of what work I have done on tasks for this research project. I have split the work up (roughly) by week, along with a separate section at the top for ongoing work/tasks.

## Current & Ongoing Tasks

- Coding: 
    * [ ] Create Python notebook for fitting GPs with variational inference based methods (EULBO)
    * [ ] Compare posterior predictive errors between iterative and batch GP algorithms
    * [ ] Compare runtime (empirical, not theoretical) between iterative and batch GP algorithms
- Theoretical:
    * [ ] Update [theory/notes document](./Notes/Research-General-Notes.tex)
- Reading List:
    * [x] *Computation-Aware Gaussian Processes* 
    * [x] *Approximation-Aware Bayesian Optimization* 
    * [x] *Variational Inference papers*
    * [x] *Posterior and Computational Uncertainty in Gaussian Processes* (<https://arxiv.org/pdf/2205.15449>)
- Other:

---

## Completed Tasks (Organized by Week)

### June 09 - June 15

- [x] Create [Variational Inference Notes](./Notes/Variational-Inference-Notes.tex)

### June 02 - June 08

- [x] Create an entropy-based acquisition function in Pytorch
- [x] Create a new notebook to compare [iterative and batch GP algorithms](./Code/IterGP-Comparison/iterative_batch_gp_comparison.ipynb)

### May 26 - June 01

- [x] Create code for 'batch' selection of action matrices
- [x] Implement other test functions (Hartmann 6D) in Pytorch 
- [x] Update notebooks to have more functions and more flexibility

### May 19 - May 25

- [x] Go over the [pytorch Gradient Descent demo](./Code/Gradient-Descent-Demo/grad_descent_example.ipynb)
- [x] Update the iterative GPs notebook to work properly with selection matrices (completed, but incorrectly)

### May 12 - May 18

- [x] Learn/relearn the general necessary Python skills needed for this project
    * [x] Relearn general Python syntax and code structure
    * [x] Relearn pandas, numpy, scipy library syntax and common commands
    * [x] Learn how to use the Pytorch (torch) library
- [x] Make further changes to the [Iterative Gaussian Processes notebook](./Code/GP-Demo-Code/iterative_gp_selection_conditioning.ipynb)
    * [x] Edit code to randomly generate "feature weight" matrices 
    * [x] Create a helper function (`normalize_selection`) to ensure the weight matrices have the same norm

### May 05 - May 11 

- [x] Create + organize Git repository for the research project
- [x] Create weekly log for task management
- [x] Complete STAT 520P problem set
- [x] Python setup
    * [x] Install and update Python
    * [x] Reinstall Anaconda and Jupyter
    * [x] Download common libraries (pandas, numpy, scipy, etc.)
- [x] Read additional Gaussian Process/Bayesian Optimization notes
- [x] Make changes to the [Iterative Gaussian Processes notebook](./Code/GP-Demo-Code/iterative_gp_conditioning_demo.ipynb)
    * [x] Make the `update_K_chol` function more efficient
    * [x] Change the iterative selection in the loop in the final cell
