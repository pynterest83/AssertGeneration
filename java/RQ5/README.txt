
This folder contains all necessary artifact for the regeneration of the Defects4J results (RQ5). This directory contains all necessary data and scripts to regenrate the results. 


TOGA: All data for regerating the TOGA results are taken from the original TOGA artifact, which can be found at https://github.com/microsoft/toga

TOGLL_prediction:
================================================ 

-   TOGLL_prediction/evosuite_reaching_tests: contains the input and meta files used by TOGA, we have also used these files as inputs for oracle generation using TOGLL.
-   TOGLL_prediction/input_data: contains the .pickle file for TOGLL oracle prediction. 
-   TOGLL_prediction/models: conatins the fine-tuned model for oracle inference. The last modification time for this model is Feb 22nd, which is before the initial submission. 
-   TOGLL_prediction/scripts: contains necessary scripts for input processing and output processing
-   TOGLL_prediction/inference: contains the scripts for TOGLL inference
-   TOGLL_prediction/results: contains the overall predicion file and oracle type wise prediction files. This directory also contains the oracle type wise test execution results which are used to generate the final results.


bug_detection:
================================================ 

- bug_detection/script: contains scripts for type wise prediction generation and result analysis scripts
- bug_detection/TOGA: contains type wise TOGA prediction results
- bug_detection/TOGLL: contains type wise TOGLL prediction results


To generate the results from Table IV and V from the paper, run the follwing commands:

To generate Table IV results:
================================================

cd TOGLL_prediction
python scripts/compute_TOGLL_metrics.py
python scripts/compute_TOGA_metrics.py


To generate Table V results:
================================================

cd bug_detection
python script/result_analysis.py


To regenerate the results from scratch, follow the below steps:
========================================================================================


Step 1: Copy the Defects4J input dataset from TOGA artifact (TOGLL_prediction/evosuite_reaching_tests) and prepare the data into a .pickle file so that our method TOGLL can consume it as inputs and generate test oracles (TOGLL_prediction/input_data)
Step 2: Generate TOGLL inference (TOGLL_prediction/models, TOGLL_prediction/inference, TOGLL_prediction/results)
Step 3: Initiate the TOGA Docker container using the following commands:

docker pull edinella/toga-artifact
docker run -i -t edinella/toga-artifact
export PATH=$PATH:/home/defects4j/framework/bin

Step 4: 

Using docker cp command copy the prediction files inside the docker container. 



For example, if you want to find the bugs that are detected by TOGLL explicit assertion oracles, run the following commands 


docker cp bug_detection/TOGLL/assertion_prefix/oracle_preds.csv  toga:/home/icse2022_artifact/eval/rq3/

Now, run the bug detection study using the command inside the docker container, 

cd /home/icse2022_artifact/eval/rq3
bash rq3.sh

This will generate a directory called toga_generated. Now copy the directory to local repository

docker cp  toga:/home/icse2022_artifact/eval/rq3/toga_generated/  bug_detection/TOGLL/assertion_prefix/togll_generated


Similarly, run the above commands for finding the bugs detected by exception oracle and prefix only. 


Step 5: 

Run the bash scripts to generate the results: 

cd bug_detection
python script/result_analysis.py
