## 1 Setup environment
Since some packages cannot be installed in the sctui/mlui we use a virtual envirorment, which is the optimal solution for using python packages.

```
pip install virtualenv
virtualenv venv
source setupROOT.sh
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## 2 BDT execution
First of all, modify the `Utils/config.yaml` so that the diccionary that you use has the variables of you input tree.

If you are runnig on CPU comment out `'tree_method':'gpu_hist'`, this option is only for GPU.
Execute it with
```
python mva_runner.py -m XGBoost -i /lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/out_2l1HadTau_Preselection/ -c /lhome/ific/p/pamarag/Work/BDT_tHq/tHqIFIC/tHqMVA/Utils/config.yaml -o output_BDT -s tHq -t tHqLoop_nominal_Loose 
```

### 2.1 Train BDT
To train the BDT comment the optimization part and the LoadModel.

### 2.2 Feature optimization
For feature optimization comment out the feature importance of the model, the train of the model, the restart to initial data, the second split and the returns. The output of this optimization is 'feature_optimize.csv', in which the most discrinant variables are in the rows below (the ones with smaller auc). This optimization does the following for each varaible:
1. Remove the varaible
2. Train the BDT without it
3. Evaluate the BDT (without the variable): roc_auc, log_loss, f1, accuracy
4. Add a row in the csv file with the matrics for the BDT when trained without this varaible

Note: The accuracy and the f1 are not the most reliable metrics when the background is much larger than the signal (as is happening for the tH analysis) because identifying everything as background will result nice accuracy.

In orther to properly detect correlation between our variables we execute `Utils/Correlation.py`. Which outpus correlation 2Dhistos for pairs of variables which have a correlation greater than 66%

### 2.3 Hyperparameter tuning
#### 2.3.1 Hyperparameter tuning traditional
There are two methods for tuning the hyperparameters.
- Loop over a vector of values for a Hyperparameter logarithmically spaced (default). Here we use the `df = RandomOptimize(m_model,opt,-4,3,50, name = m_model.signal)` where `opt` is the hyperparameter of the BDT that we want to optimze, for instance: `opt = {'min_child_weight': 0.005}`.
Usage: `df = RandomOptimize(m_model,opt, initial power of 10, final power of 10, amount of values, name = m_model.signal)`
 This method only works with parameters which are continuous, therefore, variables like  `max_depth` or `n_jobs` cannot be tuned using this method.
- Evaluates the derivative. In this method the number of iterations and the pass size have to be selected by hand. 

The output of the optimization is a csv file like 'opt_rand_min_child_weight_tHq.csv'

The first hyperparameter to optimize is the `scale_pos_weight`, the factor used to scale the positve (signal) events. A good starting point for a low Sig/Bkg ratio analysis like tHq would be (bkg_yield/sig_yield)*0.30. This parameter is tuned by hand.
Secondly, we optimize the `learning_rate`, the shrinkage factor employed to slow down the learning in the gradient boosting mode. If we don't see good results after using RandomOptimize on the learning rate, we must tune the scale_pos_weight again. Finally, the `min_child_weight` is optimized. This is an iterative process.

#### 2.3.2 Hyperparameter tuning with Genetic Algorithm 
1. Initialise Population:
2. Fitness function:
3. Selection and Drop:
4. Diversify population:
    - Cross pair
    - Mutation
5. Drop duplicate and Renew population
6. Iterate:
7. Results:

## 3 Send a Job with Condor
Before sending a job to condor we must deactivate the virtual environment with the command `deactivate`.
Edit the `sendJob.sh` and add the command that you want to send to condor.
Note that the `sendJob.sh` must be an executable (use `chmod u+x sendJob.sh` after creating the file).
Send the job with 
`condor_submit xgboost.sub`.
Use `condor_q` to check the jobs sent to condor. The errors are listed in the files errors.####.#.err, complains of the form "adding tree with no entries" are not relevant. The output is outfile.#####.#.out .

## 4 Run with Artemisa
You can save space time by tuning on a GPU instead of a CPU. To run the BDT in GPU mode activate the line 
`'tree_method':'gpu_hist'`
If you are using IFIC's ML infracstructe, Artemisa, you need to ask for computing time before executing the mva_runner.py. With the command `gpurun python mva_runner.py`, you are granted 5 minutes of computiong time to execute the mva_runner. Note that the virtual envirorment has to be activated and the setupROOT.sh sourced.
