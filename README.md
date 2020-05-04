# AMLS_II_assignment19_20
UCL_AMLS_II

## Set up

#### 1 - Install dev prereqs (use equivalent linux or windows package management)
        brew install python3.6
        brew install virtualenv
        
Example for Windows when not using any package management software and limited or no admin rights:

- Install python 3.6 from https://conda.io/miniconda.html and make sure you select 'add to path' when prompted byt the 
installation wizard
- If you failed to select 'add to path' during the installation: Go to Control Panel - User Accounts - Change my 
environment variables. Under User variables find 'Path' and click 'Edit'. Then click new and paste the path to where 
Miniconda was installed - the path can be found by opening Anaconda prompt and running:

        echo %PATH%
        
- Miniconda comes with virtual environment management so ignore the brew installations above, proceed to Section 2 and
follow Windows specific requirements.

#### 2 - Set up a Python virtual environment (from the root of the AMLS directory)
        virtualenv -p python3.6 AMLS-venv
        source AMLS-venv/bin/activate


For Windows, open cmd and run the following commands, hitting enter after each line and waiting for it to execute:  (** path-to-script ** is where you have downloaded this repository)

        cd "** path-to-script **"
        conda create -n AMLS-venv python=3.6
        activate ASMLS-venv


#### 3 - Install required python dependencies into the virtual env
        pip install -r requirements.txt

#### 4 - Run the AMLS 
 
The data file for each of the tasks are found in the Data directory. 


#### 5 - Each script is invokable individually in the command line. Follow the sample command line to run the scripts.
Run the message-polarity-classifier.py and the topic-based-message-polarity-classifier.py scripts
 separately (from inside the virtual env)
Giving: 


        postional argument:
        -i --tweets-file  The path to the TSV tweets file containing labelled tweets
                     
        Sample Invocation Command to run meassage-polarity-classifier.py 
        python /AMLS_II_assignment19_20/Task_A/message-polarity-classifier.py 
                    -i /Data/taskA_tweets.tsv
                  
        
        Sample Invocation Command to run topic-based-message-polarity.py 
        python /AMLS_II_assignment19_20/Task_B/topic-based-message-polarity-classifier.py 
                    -i /Data/taskB_tweets.tsv
     
        
        
