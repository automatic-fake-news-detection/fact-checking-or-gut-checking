
# Automatic Fake News Detection: Automatic Fake News Detection: Are current models “doing their own research” using search engines?

Ian Kelk, Wee Yi Lee, Jonathan Bourne, and Benjamin Basseri 
 <br>
1. Make sure to unzip sorted.uk.word.unigrams.zip and put the unzipped file, sorted.uk.word.unigrams., under bias folder. 
2. install a python environment with the required packages (requirements.txt) 
3. The code ran under Google colab, if you are not running from Google Colab, make sure you remove the Google Drive mounting cell.
4. Please ensure you point to the right system path directory in main.ipnyb. analyze_snes.ipynb and analyze_pomt.ipnyb. For example, in Google Colab: 
	- os.chdir('/content/drive/MyDrive/NLPProject/bias' )
	- sys.path.append('/content/drive/MyDrive/NLPProject')
	- sys.path.append('/content/drive/MyDrive/NLPProject/bias/')
	
4. When you run the code for the first time, it might take some time to download pretrained language models.
6. Run main.ipynb to generate all the results. When run in Google Colab Pro+, it will take about 70 - 100 hours to complete.
6. if you wish to select certain setup, you can change the below variables. For example:
	- steps = [['EMO_LEXI'], ['EMO_INT']]
	- modes = ['bert']
	- datasets = ['pomt']
	- inputtypes = ['CLAIM_AND_EVIDENCE', 'EVIDENCE_ONLY', 'CLAIM_ONLY']

6. After running main.ipynb, create the table and plots based on your results by running: analyze_snes.ipynb for Snopes dataset and analyze_pomt.ipnyb for PolitiFact dataset
7. To create the table and plots, if you don't have the full results, you will need to modify the below variables in analyze_snes.ipynb and analyze_pomt.ipnyb in order to run them. 
	- steps = ['none', 'neg', 'stop', 'pos', 'stem', 'all', 'pos-neg', 'pos-stop','formal', 'informal', 'pos-neg-stop', 'EMOINT', 'EMO_LEXI']
	- steps_dic = {'none':'none', 'neg':'neg', 'stop':'stop', 'pos':'pos', 'stem':'stem', 'all':'all', 
             			'pos-neg':'pos+neg', 'pos-stop':'pos+stop', 'formal':'formal', 'informal':'informal',
           			'pos-neg-stop': 'pos+neg+stop', 'EMOINT': 'Emotion intensity', 'EMO_LEXI': 'Emotion Lexicon'}
	- inputtypes = ["CLAIM_ONLY", "EVIDENCE_ONLY", "CLAIM_AND_EVIDENCE"]
	- inputtypes_dic = {"CLAIM_ONLY": "Claim", "EVIDENCE_ONLY": "Evidence", "CLAIM_AND_EVIDENCE": "Claim+Evidence"}


