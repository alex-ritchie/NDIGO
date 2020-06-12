# NDIGO
Nonparametric Density estimation of Identifiable mixture models from Grouped Observations

## Dependencies
general: numpy, sklearn, scipy, pandas, seaborn, matplotlib
topic modeling experiment: polyglot, glob, nltk, STTP (GitHub repo given in Main paper)

I have included STTP for convenience, but it is NOT my code. It is code accompanying a preprint with bibtex entry:
@article{qiang2018STTP,
  title =  {Short Text Topic Modeling Techniques, Applications, and Performance: A Survey },
  author = {Qiang, Jipeng and 
            Qian Zhenyu and
            Li, Yun and 
            Yuan, Yunhao and 
            Wu, Xindong},
  journal = {arXiv preprint arXiv:1904.07695},
  year  =  {2019}
}

## Instructions for Reproduction of Results

Scatter plots, density estimates, MAGIC figure:
run the appropriate jupyter notebook 

In sample / out of sample results on synthetic data over multiple runs
at command line run the following

python runSyntheticExps.py --dataset <dataset name> --n <number of samples> --M <number of mixture components> --nRuns <number of runs> --outOfSample <True or False>

Topic Modeling Example:
This one is a bit more involved

If you just want to see the coherence results you can run (from STTP-master)

java -jar jar/STTM.jar -model CoherenceEval -label wiki.en.text -dir <Method Here>results/ -topWords topWords

where <Method Here>results/ is a preexisting folder for each method containing .topWords files that contain the top 20 words for each topic found by a method on a given run

** Note **
I accidentally overwrote the files I used to generate the results for NDIGO, but I reran the experiments so the results are representative of (but not exactly) what was reported in the submission. The results I provide here are actually a bit better than what was shown in the paper.

If you want to run everything from scratch:

for LFDMM and GPUDMM run the following from the command line in STTM-master:
java -jar jar/STTM.jar –model <GPUDMM or LFDMM> -corpus troll-corpus.txt -vectors glove10d.txt -ntopics 10 -name <output file name>

for NDIGO:
-run the jupiter notebook Twitter Example NDIGO.ipynb 5 times with 5 different random seeds

You can then run the evaluation code as described above

