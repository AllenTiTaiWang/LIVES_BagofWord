import argparse
import process as pr
import prediction as pred
from sklearn.model_selection import train_test_split, ShuffleSplit
import plot as pltf
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

speaker = ["both", "participant", "coach"]

if __name__ == "__main__":
    """
    The main function of the analysis.

    :param mode: Necessay parameter. A string should be either 'train' or 'test'.
    :param label: Necessary parameter. A sting should be from one of the fidelity 
    measure or behavior outcome.
    :param classification: Necessary parameter. A string should be either 'logistic'
    or 'linear', depending on the param label.
    :param -e: Optional parameter.A string should be 'coach', 'participant',or 
    'both'. (Only needed on 'test' mode as a hyperparameter)
    "param -l" OPtional parameter. An integer would split the transcripts. (Only
    needed on 'test' mode as a hyperparameter)
    :param -p: Should be eiher 'head' or 'tail'. Take the first half or second (Only
    needed on 'test' mode as a hyperparameter)
    :param -s: Only needed when behavior outcome is chosed to be the label. This one
    should be the match of the first label (ex: tfat_pcal_2)
    """
    #input
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help = "coach or speaker (default: both)", type = str, dest = "people")
    parser.add_argument("mode", default = "train")
    parser.add_argument("label", help = "Prediction label", type = str, default = "availibility")
    parser.add_argument("classification", help = "logistic or linear", type = str, default = "logistic")
    parser.add_argument("-l", help = "splits of conversations", type = int, dest = "length")
    parser.add_argument("-p", help = "head or tail of conversations", type = str, dest = "part")
    parser.add_argument("-s", help = "subtract another feature to create a difference", dest = "label2")
    args = parser.parse_args()
    
    #Read Data
    if args.label2:
        table = pr.read_tables_subtract(args.label, args.label2)
        label = "difference"
    else:
        table = pr.read_tables(args.label)
        label = args.label
    
    if args.mode == "train":
        for people in speaker:
            #Process texts and label
            url_dic = pr.text_to_dic(people, table, label)
            #Train with different process with the calls
            train_full, label_res_full = pred.dic_to_feature(url_dic, 1, "head")
            train_full, test_full, label_res_full, label_test_full = train_test_split(train_full, label_res_full, test_size=0.33, random_state=42)
                
            train_head, label_res_head = pred.dic_to_feature(url_dic, 2, "head")
            train_head, test_head, label_res_head, label_test_head = train_test_split(train_head, label_res_head, test_size=0.33, random_state=42)

            train_tail, label_res_tail = pred.dic_to_feature(url_dic, 2, "tail")
            train_tail, test_tail, label_res_tail, label_test_tail = train_test_split(train_tail, label_res_tail, test_size=0.33, random_state=42)

            train = [train_full, train_head, train_tail]
            label_res = [label_res_full, label_res_head, label_res_tail]
            
            #Print the training evaluation
            print("The analysis of " + people)
            for t, l in zip(train, label_res):
                y_gold, y_pred, y_base = pred.train_and_dev(t, l, args.classification)#, args.use_string)
                pred.evaluation(args.classification, y_gold, y_pred, False)
        #Baseline
        pred.evaluation(args.classification, y_gold, y_base, True)
    
    elif args.mode == 'test':
        url_dic = pr.text_to_dic(args.people, table, label)
        train, label = pred.dic_to_feature(url_dic, args.length, args.part)
        train, test, label, label_t = train_test_split(train, label, test_size=0.33, random_state=42)
        pred.predict_test(train, label, test, label_t, args.classification)#, args.use_string)

    #PLot
    else:
        url_dic = pr.text_to_dic(args.people, table, label)
        train, label = pred.dic_to_feature(url_dic, args.length, args.part)
        #train, test, label, label_t = train_test_split(train, label, test_size=0.33, random_state=42)
        vec = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        train = vec.fit_transform(train)
        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
        
        if 'logistic' in args.classification:
            estimator = LogisticRegression(C=1000, solver='liblinear', max_iter=300, penalty='l1')
        else:
            estimator = LinearRegression()
        pltf.plot_learning_curve(estimator, "Learning Curves", train, label, axes=None, ylim=(0.1, 1.01), cv=cv, n_jobs=4)#, scoring = score)
        plt.show()
