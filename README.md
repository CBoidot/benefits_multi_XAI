# Limits of XAI evaluation #

This repo is linked to the research described in Benefits of Benefits of using multiple post-hoc explanations for Machine Learning.
You should consider reading it before using the codes here.
In a nutshell: we use a dataset of League of Legends in order to explore human-AI preformances on a binary prediction task (winner prediction).
Do XAI methods help us? Which one is the best? I investigate these questions using a multi-explanation interface for an in-subject experiment.

Here are the main app and assets to install in order to run the experiment.
Further refactoring may be done in another repo, comments are done in French, so do not consider using the files in "brouillons" (litterally "drafts") to re-compute the explanations.


### HOW TO USE ###

1/ Create an env with python 3.8

2/ install the requirements.txt in yout env

3/ correct the few problems in the libraries:
 - in shap/plots/_waterfall.py, matplotlib is imported as "pl" but some methods desperately try to call "plt"
 - in skrules/_skoperules.py, replace "from sklearn.external import six" by "import six"

4/ Pray one Pater and three Ave (optionaal)

5/ In your terminal, in the present folder, run "streamlit run main2.py"
5.1/ if it is the first time you use streamlit, you will be asked an email adress: you can skip this step.
5.2/ a window should open in your browser: **go back to your terminal and enter a number for the experiment**
5.3/ at the end of the experiment, several files will be saved in the results2/ folder.


### DISCLAIMERS ###

All the experiment is written in french.
During the experiment, you may encounter casual bugs with streamlit. Usually, adding a blank line in main2.py may save the day.
If you need more information, you may contact the author (CBoidot) directly.
