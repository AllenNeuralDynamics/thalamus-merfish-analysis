# thalamus-merfish-analysis

Code Ocean capsule for sharing analysis of thalamus MERFISH data, primarily in jupyter notebook form.


## Usage

To create your own working copy on Code Ocean, select **Duplicate** from the **Capsule** menu and use the default option ("Link to git repository"). You can then make your own changes and sync back and forth from our shared github repository (https://github.com/AllenNeuralDynamics/thalamus-merfish-analysis) via git. You can use git functions either in the **Reproducibility** panel on the right-hand side of the capsule view, or within the cloud workstation (more flexibility).

As a package: install the core functionality as a package directly from github via pip: `pip install git+https://github.com/AllenNeuralDynamics/thalamus-merfish-analysis`

As a streamlit app: start a Code Ocean cloud workstation in streamlit mode (or from the terminal on another cloud workstation, run `streamlit run /code/streamlit_app.py`)

Locally: you can clone directly from github to your personal machine, but data assets will not be available. Code Ocean also has an option to download the capsule with data assets, but this will be very large if you're not careful!

### Data assets

We are documenting the data assets used by the capsule by including their name and ID in `environment/data_assets.yml`. If a data asset is not broadly used, but required for a specific notebook or use case, include the entry commented out with an explanatory comment above. The listed data assets can then be attached from a cloud workstation using the API, by running `python environment/attach_data_assets.py` (ideally this could run in postInstall, but that's not currently possible). 

To use this API call though, you'll need to generate and attach a CO API key (one time only): 
1. Go to https://codeocean.allenneuraldynamics.org/account/apiKeys to create an access token with capsule read/write permissions.
2. Copy the token and switch to the capsule environment manager, where you'll find "Add New API Credentials" under the "Codeocean API" entry.
3. Paste the token in the "API Secret" field. The "Short Description" and "API Key" won't matter here, but you may want to use the key name from step 1 so you can refer back to it later

### Git details 

We'll generally plan to use a single branch, but feel free to create branches as needed for your own work, and merge back to the master branch to share. **Do not** attempt to "force push" in git (`git push -f`), as this seems to be incompatible with how Code Ocean interfaces with git. 

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
