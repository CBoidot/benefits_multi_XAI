# Limits of XAI evaluation #

This repo is linked to the research described in [my-article-name].
You should consider reading it before using the codes here.
In a nutshell: we use a dataset of League of Legends in order to explore human-AI preformances on a binary prediction task (winner prediction).
Do XAI methods help us? Which one is the best? I investigate these questions using a multi-explanation interface for an in-subject experiment.

Here are the main app and assets to install in order to run the experiment.
Further refactoring may be done in another repo, comments are done in French, so do not consider using the files in "brouillons" (litterally "drafts") to re-compute the explainations.
à l'utilisation d'un dataset de League of Legends, en vu d'évaluer l'efficacité des méthodes d'XAI dans l'aide à la prise de décision.

> main2.py à éxécuter pour l'expé 2.


### HOW TO USE ###

1/ Create an env (python==3.8.12)

2/ install the requiements.txt in the env

3/ correct the few problems in the libraries:
 - in shap/plots/_waterfall.py, matplotlib is imported as "pl" but some methods desperately try to call "plt"
 - in skrules/_skoperules.py, replace "from sklearn.external import six" by "import six"

4/ Pray one Pater and three Ave

5/ In your terminal, in the present folder, run "streamlit run main2.py"
 —> You may encounter casual bug. Usually, adding a blank line in main2.py may save the day. IDK
