package ude.student.fadu

import weka.attributeSelection.GreedyStepwise
import weka.classifiers.AbstractClassifier
import weka.classifiers.bayes.BayesNet
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.bayes.net.search.global.GeneticSearch
import weka.classifiers.bayes.net.search.global.HillClimber
import weka.classifiers.bayes.net.search.global.SimulatedAnnealing
import weka.classifiers.functions.*
import weka.classifiers.functions.SGD.*
import weka.classifiers.lazy.IBk
import weka.classifiers.lazy.KStar
import weka.classifiers.meta.RandomSubSpace
import weka.classifiers.rules.*
import weka.classifiers.trees.*
import weka.core.SelectedTag

@Suppress("unused")
enum class Classifier(val title: String, val description: String) {
    BAYES_NETWORK_K2(
        "BN: K2",
        "Bayes Network using K2 learning algorithm: hill climbing restricted by an order on the variables."
    ) {
        override val model get() = BayesNet()
    },
    BAYES_NETWORK_GENETIC(
        "BN: Genetic",
        "Bayes Network using genetic search learning algorithm: allow a population of Bayes network " +
                "structures to mutate and apply cross over to get offspring. The best network structure found " +
                "during the process is returned."
    ) {
        override val model get() = BayesNet().apply { searchAlgorithm = GeneticSearch() }
    },
    BAYES_NETWORK_SIMULATED_ANNEALING(
        "BN: Sim Anneal",
        "Bayes Network using the general purpose search method of simulated annealing as learning algorithm."
    ) {
        override val model get() = BayesNet().apply { searchAlgorithm = SimulatedAnnealing() }
    },
    BAYES_NETWORK_HILL_CLIMBER(
        "BN: Hill Climb",
        "Bayes Network using a hill climbing algorithm for adding, deleting and reversing arcs. The " +
                "search is not restricted by an order on the variables (unlike K2)."
    ) {
        override val model get() = BayesNet().apply { searchAlgorithm = HillClimber() }
    },
    DECISION_TABLE_BEST_FIRST(
        "DT: Best First",
        "Simple decision table majority classifier. Searches the space of attribute subsets by greedy " +
                "hill climbing augmented with a backtracking facility."
    ) {
        override val model get() = DecisionTable()
    },
    DECISION_TABLE_GREEDY_STEPWISE(
        "DT: Greedy Step",
        "Simple decision table majority classifier. Performs a greedy forward or backward search through " +
                "the space of attribute subsets."
    ) {
        override val model get() = DecisionTable().apply { search = GreedyStepwise() }
    },
    HOEFFDING_TREE(
        "Hoeffding Tree", "Hoeffding tree (VFDT) is an incremental, anytime decision tree " +
                "induction algorithm."
    ) {
        override val model get() = HoeffdingTree()
    },
    KNN_AUTO_SELECT_K(
        "KNN", "K-Nearest Neighbors, whereby hold-one-out cross-validation will be used to select " +
                "the best k value."
    ) {
        override val model get() = IBk().apply { crossValidate = true }
    },
    C4_5("C4.5", "C4.5 decision tree induction algorithm.") {
        override val model get() = J48()
    },
    RIPPER(
        "RIPPER",
        "Propositional rule learner, Repeated Incremental Pruning to Produce Error Reduction (RIPPER), " +
                "which was proposed by William W. Cohen as an optimized version of IREP."
    ) {
        override val model get() = JRip()
    },
    K_STAR(
        "K*",
        "K* is an instance-based classifier, based upon similarity function. It differs " +
                "from other instance-based learners in that it uses an entropy-based distance function."
    ) {
        override val model get() = KStar()
    },
    LMT(
        "LMT",
        "Logistic Model Trees (LMT) are classification trees with logistic regression functions at the " +
                "leaves. The algorithm can deal with binary and multi-class target variables, numeric and nominal " +
                "attributes and missing values."
    ) {
        override val model get() = LMT()
    },
    LOGISTIC("Log. Regression", "Multinomial logistic regression model with a ridge estimator.") {
        override val model get() = Logistic()
    },
    MULTILAYER_PERCEPTRON(
        "MLP",
        "Multilayer Perceptron: uses backpropagation to learn. Built automatic by a heuristic."
    ) {
        override val model get() = MultilayerPerceptron()
    },
    NAIVE_BAYES(
        "Naive Bayes",
        "Naive Bayes classifier using estimator classes. Numeric estimator precision values are chosen " +
                "based on analysis of the training data."
    ) {
        override val model get() = NaiveBayes()
    },
    ONE_R(
        "1R",
        "1R classifier: uses the minimum-error attribute for prediction, discretizing numeric attributes."
    ) {
        override val model get() = OneR()
    },
    PART(
        "PART",
        "PART decision list. Uses separate-and-conquer. Builds a partial C4.5 decision tree in each " +
                "iteration and makes the \"best\" leaf into a rule."
    ) {
        override val model get() = PART()
    },
    RANDOM_FOREST("Random Forest", "Class for constructing a forest of random trees.") {
        override val model get() = RandomForest()
    },
    RANDOM_SUBSPACE(
        "Random Subspace",
        "Decision tree based classifier. Maintains highest accuracy and improves on generalization " +
                "accuracy as it grows in complexity. Consists of trees constructed in randomly chosen subspaces."
    ) {
        override val model get() = RandomSubSpace()
    },
    RANDOM_TREE(
        "Random Tree",
        "Class for constructing a tree that considers K randomly chosen attributes at each node. " +
                "Performs no pruning."
    ) {
        override val model get() = RandomTree()
    },
    REP_TREE(
        "REPTree",
        "Builds a decision/regression tree using information gain/variance and prunes it using " +
                "reduced-error pruning (with backfitting). Missing values are dealt with by splitting the " +
                "corresponding instances into pieces (i.e. as in C4.5). "
    ) {
        override val model get() = REPTree()
    },
    SGD_HINGE(
        "SGD: Hinge",
        "Stochastic Gradient Descent for linear model learning based on Hinge loss function. Replaces " +
                "missing values, transforms nominal attributes into binary ones and normalizes attributes."
    ) {
        override val model get() = SGD().apply { lossFunction = SelectedTag(HINGE, TAGS_SELECTION) }
    },
    SGD_LOG_LOSS(
        "SGD: Log",
        "Stochastic Gradient Descent for linear model learning based on log loss function. Replaces " +
                "missing values, transforms nominal attributes into binary ones and normalizes attributes."
    ) {
        override val model get() = SGD().apply { lossFunction = SelectedTag(LOGLOSS, TAGS_SELECTION) }
    },
    SIMPLE_LOGISTIC(
        "Simple Logistic",
        "Linear Logistic Regression: LogitBoost with simple regression functions as base learners is used " +
                "for fitting the logistic models. Optimal number of LogitBoost iterations through cross-validation " +
                "leading to automatic attribute selection."
    ) {
        override val model get() = SimpleLogistic()
    },
    SMO(
        "SMO",
        "Implements John Platt's Sequential Minimal Optimization (SMO) algorithm for training a " +
                "support vector classifier"
    ) {
        override val model get() = SMO()
    },
    VOTED_PERCEPTRON(
        "VotedPerceptron", "Implementation of the voted perceptron algorithm by Freund and Schapire. " +
                "Globally replaces all missing values, and transforms nominal attributes into binary ones."
    ) {
        override val model get() = VotedPerceptron()
    },
    ZERO_R(
        "0-R",
        "0-R classifier. Predicts the mean (for a numeric class) or the mode (for a nominal class)."
    ) {
        override val model get() = ZeroR()
    };

    abstract val model: AbstractClassifier
    val additionalInformation
        get() = try {
            val method = model.javaClass.getMethod("globalInfo")
            if (method.returnType == String::class.java) {
                method.invoke(this) as String
            } else "-"
        } catch (e: NoSuchMethodException) {
            "-"
        }
}