import pandas as pd
import numpy as np
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.exceptions import H2OValueError
from h2o.tree import H2OTree

class H2ORuleFit():
    """
    H2O RuleFit
    Builds a Distributed RuleFit model on a parsed dataset, for regression or 
    classification. 
    :param min_depth: Minimum length of rules. Defaults to 1.
    :param max_depth: Maximum length of rules. Defaults to 10.
    :param nfolds: Number of folds for K-fold cross-validation. Defaults to 5.
    :param seed: Seed for pseudo random number generator. Defaults to -1.
    :returns: a set of rules and coefficients
    :examples:
    >>> rulefit = H2ORuleFit()
    >>> training_data = h2o.import_file("smalldata/gbm_test/titanic.csv", 
    ...                                  col_types = {'pclass': "enum", 'survived': "enum"})
    >>> x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
    >>> rulefit.train(x=x,y="survived",training_frame=training_data)
    >>> rulefit
    """
    
    def __init__(self, min_depth=1, max_depth=10, nfolds=5, seed=-1):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.nfolds = nfolds
        self.seed = seed
        
    def train(self, x=None, y=None, training_frame=None, offset_column=None, fold_column=None, weights_column=None,
              validation_frame=None, **params):
        """
        Train the rulfit model.
        :param x: A list of column names or indices indicating the predictor columns.
        :param y: An index or a column name indicating the response column.
        :param training_frame: The H2OFrame having the columns indicated by x and y (as well as any
            additional columns specified by fold, offset, and weights).
        :examples:
        >>> rulefit = H2ORuleFit()
        >>> training_data = h2o.import_file("smalldata/gbm_test/titanic.csv", 
        ...                                  col_types = {'pclass': "enum", 'survived': "enum"})
        >>> x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
        >>> rulefit.train(x=x,y="survived",training_frame=training_data)
        >>> rulefit
        """
        family = "gaussian"
        if (training_frame.type(y) == "enum"):
            if training_frame[y].unique().nrow > 2:
                raise H2OValueError("Multinomial not supported")
            else:
                family = "binomial"


        # Get paths from random forest models
        paths_frame = training_frame[y]
        depths = range(self.min_depth, self.max_depth + 1)
        rf_models = []
        for model_idx in range(len(depths)):

            # Train random forest models
            rf_model = H2ORandomForestEstimator(seed = self.seed, 
                                                model_id = "rf.hex", 
                                                max_depth = depths[model_idx])
            rf_model.train(y = y, x = x, training_frame = training_frame)
            rf_models = rf_models + [rf_model]

            paths = rf_model.predict_leaf_node_assignment(training_frame)
            paths.col_names = ["rf_" + str(model_idx) +"."+ x for x in paths.col_names]
            paths_frame = paths_frame.cbind(paths)

        # Extract important paths
        glm = H2OGeneralizedLinearEstimator(model_id = "glm.hex", 
                                            nfolds = self.nfolds, 
                                            seed = self.seed,
                                            family = family,
                                            alpha = 1, 
                                            remove_collinear_columns=True,
                                            lambda_search = True)
        glm.train(y = y, training_frame=paths_frame)

        intercept, rule_importance = _get_glm_coeffs(glm)
        rule_importance = pd.DataFrame.from_dict(rule_importance, orient = "index").reset_index()
        rule_importance.columns = ["variable", "coefficient"]

        # Convert paths to rules
        rules = []
        for i in rule_importance.variable:
            if family == "binomial":
                model_num, tree_num, path = i.replace("rf_", "").replace("T", "").replace("C1.", "").split(".")
            else:
                model_num, tree_num, path = i.replace("rf_", "").replace("T", "").split(".")
            tree = H2OTree(rf_models[int(model_num)], int(tree_num)-1)
            rules = rules + [_tree_traverser(tree.root_node, path)]

        # Add rules and order by absolute coefficient
        rule_importance["rule"] = rules
        rule_importance["abs_coefficient"] = rule_importance["coefficient"].abs()
        rule_importance = rule_importance.loc[rule_importance.groupby(["rule"])["abs_coefficient"].idxmax()]  
        rule_importance = rule_importance.sort_values(by = "abs_coefficient", ascending = False)
        rule_importance = rule_importance.drop("abs_coefficient", axis = 1)
        
        self.intercept = intercept
        self.rule_importance = rule_importance
        


def _get_glm_coeffs(glm):
    """
    Get the GLM coefficients by choosing the lambda with diminishing returns on explained deviance
    """
    r = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(glm)
    deviance = r.get('explained_deviance_train')
    inflection_pt = [i*3 for i, x in enumerate(np.diff(np.sign(np.diff(deviance, 2)))) if x != 0 and i > 0][0]
    intercept = {k: v for k,v in r.get('coefficients')[inflection_pt].items() if  k == "Intercept"}
    coeffs = {k: v for k,v in r.get('coefficients')[inflection_pt].items() if abs(v) > 0 and k != "Intercept"}
    return intercept, coeffs


def _tree_traverser(node, split_path):
    """
    Traverse the tree to get the rules for a specific split_path
    """
    rule = []
    splits = [char for char in split_path]
    for i in splits:
        if i == "R":
            if np.isnan(node.threshold):
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.right_levels, 
                                'operator': 'in'}]
            else:
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.threshold, 
                                'operator': '>='}]

            node = node.right_child
        if i == "L":
            if np.isnan(node.threshold):
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.left_levels, 
                                'operator': 'in'}]

            else:
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.threshold, 
                                'operator': '<'}]

            node = node.left_child
    consolidated_rules = _consolidate_rules(rule)
    consolidated_rules = " AND ".join(consolidated_rules.values())
    return consolidated_rules

def _consolidate_rules(rules):
    """
    Consolidate rules to remove redundancies
    """
    rules = [x for x in rules if x.get("value")]
    features = set([x.get('split_feature') for x in rules])
    consolidated_rules = {}
    for i in features:
        feature_rules = [x for x in rules if x.get('split_feature') == i]
        if feature_rules[0].get('operator') == 'in':
            cleaned_rules = i + " is in " + ", ".join(sum([x.get('value') for x in feature_rules], []))
        else:
            cleaned_rules = []
            operators = set([x.get('operator') for x in feature_rules])
            for op in operators:
                vals = [x.get('value') for x in feature_rules if x.get('operator') == op]
                if '>' in op:
                    constraint = max(vals)
                else:
                    constraint = min(vals)
                cleaned_rules = " and ".join([op + " " + str(round(constraint, 3))])
                cleaned_rules = i + " " + cleaned_rules
        consolidated_rules[i] = cleaned_rules
    
    return consolidated_rules
