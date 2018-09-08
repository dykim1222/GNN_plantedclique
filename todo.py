'''
##########################    TASKS TO DO:    ###########################

3. try resnet/preac-resnet
3. try different sparsification

4. change loss function...? sensitivity/imbalanced labels/ f1-score?

***
mos: change loss and output structure
5. choose M
5. resolve catastrophic forgetting if any; by thinking about forgetting later with RNN structure.
***



:::Hyperparameters:::
%%% MODEL
0. graph operators
3. normalization

4. num_features
5. num_layers

%%% OPTIM
0. lr scheduler

%%% Data Generation
0. data distribution
6. tt
7. ss

%%% Done but maybe again later
1. optimizer                rmsprop
1. learning_rate            0.001
2. init                     stick with base first
4. activation function      relu

--Exp 1.--
    %%% Baseline: tt=0.9, adam, delta, nf=10, nl=30
    1. 0.7 Line GNN needs more train. Too slow.. give it more time
    2. nf up/down: nf=6: not that bad but little worse; nf=20: seems little better but negligible
    2.1. Weird thing: baseline learns bell curve in the testing time even if track_running_stats is off..
    3. Don't do 0.6 directly.
    4. 0.8 performs the same as or little bit better than 0.9
    5. 0.7 seems to do little better than 0.9 in terms of pulling but worse in the tail
    6. SET tt=0.8 as default!

--Exp 2.--
    %%% Baseline: tt=0.8, adam 0.001, delta, nf=100, nl=60 ... <=SOTA
    0. track  NOooooooooo. Never.
    1. optim: to do next
        - adadelta: default (little better than adam), smaller (too slow)
        - adagrad : default (somewhat better than adam), smaller (little better than adam)
        - adamax  : same
        - rmsprop : try both
        - adam    : same
        - sgd     : try
    2. init: stick with base until later.
    3. num_layers: no more than 60
    4. learning rate down--> by 10^-1

--Exp 3.--
    %%% Baseline: tt=0.8, adam 0.001, delta, nf=100, nl=60
    1. 0.8 line go with M=2000 and 3000: still running.
    2. optim:   rmsprop 0.001 ( later adagrad was also ok.)
    3. densenet: implement!
    4. layer norm, instance norm: bad. use BN.

--Exp 4.-- optim still adam 0.001
    1. run bp with 1000 test examples.
    2. densenet: nf(3) X nl(3) X md(4) X reduc(2) = 72 + 4(track on)   # nf divisible by 16? (some are killed due to out of memory...)
    3. Use the available info in the test evaluation (the size of the clique): re-run test. change test eval measure::: changed.

--Exp. 5.-- optim now rmsprop 0.001
    1. re-run densenets: nf(2) X nl(2) X md(3) X reduc(2) = 24
        - run with small layers first... same performance yet..
        - keep doing more features
        - reduction 1/2 is better than reduction 1.0
        - maybe try different transition layers

    3. activation function
        - just try relu... but still more to check

--Exp. 6.--  nr --num_features 16, 48 --tt 0.7,0.6 --
    1. try different num_features(16,48), tt(0.6,0.7)
        - tt 0.6 is super bad.
        - tt 0.7 is bad.
        - num_features: no performace difference in this region at least.
    2. try different transition layers
        - inter_factor 10 >> 30 >> 4:: now set default to 10--> find the best inter_factor
    2. try curriculum
        - not so helpful
--Exp. 7.--
    1. weight on BCEWithLogitsLoss.
        - weight boost 10 > 30 > 1.

--Exp. 8.--
    1. try diff tt: 0.766, 0.733
        - not so bad as expected. more to explore here. seems like .766 performs the best. set as default
    2. for tt=0.8, try nf 48, 32
        - not so good, yet. more to see. seems like more features helps performance.

--Exp. 9.--
    1. deeper layer 100
        - not helpful. still waiting....

--Exp. 10.-- Back in NYC..!
    1. Plain & Line GNN for N=1000, 2000
        - for N=1000, not much difference, but line gnn more stable  ==> keep running line gnn 1000 ==> no difference in performance
        - for N=2000, need more resource. only 7 layers and performace was not as expected ==> Ask Bruna for any more powerful resource
    2. Densenet for N=2000
        - still perform better than BP as in N=1000 case, but BP is catching up.. ==> run more dense 2000
        - dense still performing better than BP ~1.5 better! :D Good.

--Exp. 11.-- Back from burn-out!
    1. Run 2-phase GNN model.

'''
