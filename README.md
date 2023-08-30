# thalamus-merfish-analysis

Code Ocean capsule for sharing analysis of thalamus MERFISH data, primarily in jupyter notebook form.


## Usage

To create your own working copy on Code Ocean, select **Duplicate** from the **Capsule** menu and use the default option ("Link to git repository"). You can then make your own changes and sync back and forth from our shared github repository (https://github.com/AllenNeuralDynamics/thalamus-merfish-analysis) via git. You can use git functions either in the **Reproducibility** panel on the right-hand side of the capsule view, or within the cloud workstation (more flexibility).

Feel free to make minor environment changes, but make sure they work by testing in a cloud workstation before you commit/push. If you're making a dramatic change (installing something with a ton of dependencies) that may be an indication you should create a separate capsule for that work.

You can also work locally by cloning directly from github to your personal machine, although this is less ideal for sharing as you'll be working with a different environment and different data paths. These issues could perhaps be minimized in the future by using environment variables to store alternative paths etc for the local vs CO contexts if this functionality seems important.

### Git details 

We'll generally plan to use a single branch, but feel free to create branches as needed for your own work, and merge back to the master branch to share. **Do not** attempt to "force push" in git (`git push -f`), as this seems to be incompatible with how Code Ocean interfaces with git. 

# Code Ocean basics

## Code, Data, Results
Your capsule starts with the folders `/code`, `/data`, and `/results`. See our help article on [Paths](https://help.codeocean.com/getting-started/uploading-code-and-data/paths) for more information.

You can upload files to the `/code` or `/data` folders using the buttons at the top of the left-side **Files pane**.

Any plots, figures, and results data should be saved in the `/results` directory. At the end of each run, these files will appear in the right-side **Reproducibility pane**, where you can view and download them. When you publish a capsule, the most recent set of results will be preserved as part of the capsule.

## Environment and Dependencies

Click **Environment** on the left to find a computational environment to accommodate your software (languages, frameworks) or hardware (GPU) requirements. You can then further customize the environment by installing additional packages. The changes you make will be applied the next time your capsule runs. See our help articles on [the computational environment](https://help.codeocean.com/getting-started/the-computational-environment/configuring-your-computational-environment-an-overview) for more information.

### Environment Caching

The next time you run your capsule after making changes to any part of the environment, a custom environment will be built and cached for future runs.

When you publish a capsule, its computational environment will be preserved with it, thereby ensuring computational reproducibility.

## Troubleshooting

If you run into any issues or have any questions, we're here to help. You can reach out to us on on live chat, or, if you prefer, write to [support@codeocean.com](mailto:support@codeocean.com).

Use the **Help** menu above to explore the different sections of our [knowledge base](https://help.codeocean.com), with answers to many common questions.
