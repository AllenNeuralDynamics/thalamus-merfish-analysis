# thalamus-merfish-analysis

Code Ocean capsule for sharing analysis of thalamus MERFISH data, primarily in jupyter notebook form.

## Usage

To create your own working copy, select **Duplicate** from the **Capsule** menu and use the default option ("Link to git repository"). You can then make your own changes and sync back and forth from our shared github repository via git (https://github.com/AllenNeuralDynamics/thalamus-merfish-analysis.git).

Feel free to make minor environment changes, but make sure they work by testing in a cloud workstation before you commit/push. If you're making a dramatic change (installing something with a ton of dependencies) that may be an indication you should create a separate capsule for that work.

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
